"""Render the §2.5 inspection report (task group 6, W-234; calibrated rerun, W-249) from saved outputs.

Reads an evaluation directory written by preprocessing_synthetic.py
(manifest.json, tables/ and curves/) and writes one standalone HTML file with
inline figures and styles. It supports human review; it is not scientific
acceptance and gives no recommended defaults. The manifest schema selects the
layout: a W-233 evaluation gives the W-234 inspection report; a calibrated
evaluation (W-248) gives the report of item 6 of the accepted amendment in
docs/preprocessing-algorithms.md: a human summary first, then the method
sections with the nine visualization changes, then an appendix.

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
sweep or evaluation is rerun. The calibrated report's detection overlays run
detection once per displayed image at the saved development-selected
threshold; the spot, match and read counts and the cutoffs must equal the
saved curve row, or rendering stops.

    uv run python ../../benchmarks/preprocessing_report.py --evaluation <run>/evaluation \
        --output <run dir>/report.html [--summary <run dir>/render-summary.json]
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
import re
import subprocess
import sys
import tempfile
import warnings

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


def check_revision(software, repo=ROOT, sources=EVALUATION_SOURCES):
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
    changed = _git(repo, "diff", "--name-only", evaluated, "--", *sources).split()
    changed += _git(repo, "ls-files", "--others", "--exclude-standard", "--", *sources).split()
    if changed:
        raise RevisionMismatch(f"evaluation sources changed since {evaluated}: {', '.join(sorted(changed))}")
    return dict(manifest_revision=revision, dirty=bool(software.get("dirty")),
                uncommitted_diff_sha256=software.get("uncommitted_diff_sha256"), evaluated_revision=evaluated,
                match=how, sources_unchanged=list(sources))


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
            try:
                self._cache[relative] = pd.read_csv(self.evaluation / relative)
            except pd.errors.EmptyDataError:  # an empty table is saved as a bare newline
                self._cache[relative] = pd.DataFrame()
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
    """Check the saved outputs and write the report of the manifest's design; returns a summary dict."""
    schema = json.loads((Path(evaluation) / "manifest.json").read_bytes()).get("schema")
    if schema == harness.CALIBRATED_SCHEMA:
        return render_calibrated(evaluation, output, repo=repo, command=command)
    return render_w233(evaluation, output, repo=repo, command=command)


def render_w233(evaluation, output, *, repo=ROOT, command=None):
    """Check the saved W-233 outputs and write the W-234 inspection report; returns a summary dict."""
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


# --- Calibrated rerun report (W-249) -------------------------------------------------------------------
#
# Item 6 of "Evaluation design amendment for the calibrated rerun (Accepted, W-243)" in
# docs/preprocessing-algorithms.md: the human summary first (setup, one card per method, cross-cutting
# findings, reading order), then the method sections with the nine visualization changes, then the
# appendix (identity, checksums, full tables). The W-234 checks run before anything is written.

CALIBRATED_REPORT_SCHEMA = "starfinder.benchmark.preprocessing_report.calibrated/1"
#: The calibrated tables also depend on the W-238 statistics tool.
CALIBRATED_SOURCES = EVALUATION_SOURCES + ("benchmarks/image_statistics.py",)
#: Change 9: the displayed scenes are held-out seed 100, in uint8 (the measured scale).
DISPLAY_SEED = harness.EVAL_SEEDS[0]
DISPLAY_DTYPE = "uint8"
MODES = tuple(harness.THRESHOLD_MODES)
#: Line style of each threshold mode in every figure.
MODE_STYLE = {"noise": dict(color="tab:blue", linestyle="-"), "adaptive": dict(color="tab:purple", linestyle="-.")}
#: Change 9: crop half-width in Y and X around the dim punctum; Z keeps all planes.
CROP_YX = 12
ROLE_ORDER = ("targeted", "harm_check", "combined", "reported", "harm_test")
ROLE_LABEL = {"targeted": "targeted", "harm_check": "clean harm check", "combined": "combined, reported",
              "reported": "reported outside the flag", "harm_test": "harm test (C7)"}
ENDPOINT_LABEL = {"max_f1": "max-F1", "correct_fraction": "correct-decode fraction"}

#: One sentence per calibrated condition and multi-FOV set.
CALIBRATED_TEXT = {
    "clean": "The calibrated baseline: puncta with mild channel gains and round trend, a uniform pedestal of 5 grey "
             "levels, and Poisson, white and spatially correlated noise; the harm check of every single-FOV method.",
    "background_only": "clean plus the per-channel offsets, the X–Z gradient, three tissue regions and texture blobs.",
    "round_effect_only": "clean plus the readout factors: stronger channel gains and round trend, weakening and "
                         "crosstalk.",
    "combined": "clean plus every readout and background factor, without geometry; reported beside each comparison "
                "and outside the flag.",
    "baseline": "clean plus per-channel offsets of 0 to 1.5 grey levels that rotate by one channel per round.",
    "gradient": "clean plus a linear background ramp along X and Z.",
    "regions": "clean plus three Gaussian tissue regions of height 2 to 3 grey levels, weighted per channel.",
    "texture": "clean plus eight texture blobs, dimmer than the puncta.",
    "gain": "clean plus the calibrated channel factors 1.08, 1.02, 0.98 and 0.93 (W-241).",
    "trend": "clean plus a stronger round trend (0.963 × 0.96 per round).",
    "gain_baseline": "gain and baseline together, a target of the pre-normalization extraction source.",
    "gain_texture": "gain and texture together, a target of the pre-normalization extraction source.",
    "combined_geometry": "combined plus a translation and a local deformation, processed with translation "
                         "registration.",
    "bright_outliers": "clean plus four blobs at 4 × the brightness median (352 grey levels in uint8), whose cores "
                       "clip at 255 in uint8 (C8).",
    "saturation": "clean with every intensity parameter multiplied by k, the smallest 2^(j/4) whose mean clipped "
                  "fraction on the development seeds reaches 10⁻³ (C1).",
    "gain_strong": "clean with channel gains spread to the largest measured LN spread of 3.15 (C6).",
    "clean_unbalanced": "clean with the unbalanced codebook, whose rounds use the four colours 7, 5, 3 and 1 times; "
                        "histogram matching's harm test (C7).",
    "mf_density": "Three FOVs on combined with 80, 20 and 2 amplicons (dense, sparse and near-empty), the targeted "
                  "set of sample-level fitting.",
    "mf_gain_drift": "Three FOVs of 80 amplicons on combined with readout gains × 1, 0.75 and 0.5, the secondary "
                     "sample-level target.",
    "mf_density_clean": "The density set on clean, the harm check of sample-level fitting.",
}

#: Method cards in the order of the page's comparison table. rows: the displayed conditions (the targeted one
#: shown passes its precondition in some mode where such a condition exists); multi-FOV rows are (set, FOV role).
#: direct: the direct metrics that apply (identical in both modes); mode_direct: direct metrics per mode.
CARDS = {
    "scalar_background": dict(
        title="Scalar background subtraction",
        does="Subtracts one level per channel and round, the channel's 10th percentile "
             "(ScalarBackgroundConfig(percentile=10.0)), and clips at 0.",
        problem="A constant additive offset per channel and round biases the channel ratios used for colour calling.",
        comparison="isolated", rows=("baseline", "clean", "combined"), background=True,
        direct=("bg_rmse", "bg_bias", "contrast_median", "clipped_fraction"), mode_direct=()),
    "background_3d": dict(
        title="3D background subtraction",
        does="A volumetric white top-hat: grey opening with an ellipsoidal footprint of radius (3, 5, 5) ZYX voxels, "
             "subtracted from the image.",
        problem="Spatially varying background that also changes along Z; slice-wise XY methods ignore Z sampling.",
        comparison="isolated", rows=("gradient", "clean", "combined"), background=True,
        direct=("bg_rmse", "bg_bias", "contrast_median", "clipped_fraction"), mode_direct=(),
        caveat="r_z is capped at 3 on the 8-plane scenes (C4), a stated deviation from ceil(3σ) + 1 = 6."),
    "percentile_normalization": dict(
        title="Percentile normalization",
        does="Maps each channel's 1st to 99.9th percentile range linearly onto the dtype range and keeps the input "
             "dtype (PercentileNormalizationConfig(p_low=1, p_high=99.9)).",
        problem="Channel and round gain differences, robust to the bright outliers that set min–max's scale.",
        comparison="isolated", rows=("bright_outliers", "clean", "combined"),
        direct=("intensity_cv", "contrast_median", "clipped_fraction", "saturated_fraction"), mode_direct=(),
        caveat="bright_outliers replaces the bright texture blobs as the outlier target (item 5); about 0.1 % of "
               "voxels saturate by design at p_high = 99.9."),
    "sample_level_fitting": dict(
        title="Sample-level fitting (mode)",
        does="Fits each recipe's statistics once over the three FOVs of a set (fit=\"supplied\") instead of per FOV "
             "(fit=\"fov\").",
        problem="Per-FOV statistics depend on content: in a near-empty FOV the high percentile falls in the noise, "
                "so normalization stretches noise over the full range.",
        comparison="ablation_r2_scalar", rows=(("mf_density", "near_empty"), ("mf_gain_drift", "gain_0.50"),
                                               ("mf_density_clean", "near_empty")),
        direct=(), mode_direct=("per_fov_correct_fraction_range",),
        caveat="Both multi-FOV precondition checks are reported (C5): the general check tests the calibrated "
               "combined base of the set; a set counts for a recipe's flag only if the set-specific check of that "
               "recipe's per-FOV-fitted arm (near-empty or gain-0.50 FOV against the dense or gain-1.00 FOV) also "
               "passes. The sets have no separate combined condition and run in uint8 only (C2)."),
    "extraction_source": dict(
        title="Pre-normalization extraction source (mode)",
        does="Extracts the colour-calling intensities from the background-corrected snapshot before normalization "
             "(extraction_source=\"bg_corrected\") instead of from the detection image.",
        problem="Normalization changes the intensities used for colour calling; the snapshot keeps linear channel "
                "ratios.",
        comparison="ablation_scalar", rows=("saturation", "clean", "combined"), snapshot=True,
        direct=("intensity_cv", "saturated_fraction"), mode_direct=("color_call_agreement",),
        caveat="Both arms detect on the same image, so detections are identical and only extraction differs; the "
               "panels compare the image each arm extracts from. saturation joins the targets (item 5)."),
    "min_max_normalization": dict(
        title="Min–max normalization (baseline method)",
        does="Maps each channel's minimum to maximum onto 0–255 and forces uint8 output.",
        problem="Channel and round gain differences.",
        comparison="isolated", rows=("gain_strong", "clean", "combined"),
        direct=("intensity_cv", "contrast_median", "saturated_fraction"), mode_direct=(),
        caveat="bright_outliers is reported beside the flag, outside it (item 5)."),
    "histogram_matching": dict(
        title="Histogram matching (baseline method)",
        does="Matches every channel's intensity distribution to a reference distribution.",
        problem="Channel intensity distributions that differ in shape; it assumes barcode bases are balanced across "
                "channels.",
        comparison="isolated", rows=("gain_strong", "clean", "combined", "clean_unbalanced"),
        direct=("intensity_cv", "contrast_median", "saturated_fraction"), mode_direct=(),
        caveat="The unbalanced-codebook harm test is reported beside the flag, not in it (C7); bright_outliers is "
               "reported outside the flag."),
    "reconstruction": dict(
        title="XY morphological reconstruction (baseline method)",
        does="Recipe 1's background step: slice-wise XY morphological reconstruction, subtracted from the image.",
        problem="Spatial background in XY.",
        comparison="isolated", rows=("regions", "clean", "combined"), background=True,
        direct=("bg_rmse", "bg_bias", "contrast_median", "clipped_fraction"), mode_direct=()),
    "white_tophat": dict(
        title="XY white top-hat (baseline method)",
        does="Slice-wise XY white top-hat with radii in pixels; it has only the isolated comparison.",
        problem="Spatial background in XY.",
        comparison="isolated", rows=("regions", "clean", "combined"), background=True,
        direct=("bg_rmse", "bg_bias", "contrast_median", "clipped_fraction"), mode_direct=()),
}

DIRECT_LABEL = {"bg_rmse": "background RMSE", "bg_bias": "background bias", "contrast_median": "puncta contrast",
                "intensity_cv": "true-spot intensity CV", "clipped_fraction": "clipped fraction",
                "saturated_fraction": "saturated fraction", "color_call_agreement": "colour-call agreement",
                "per_fov_correct_fraction_range": "per-FOV correct-decode range"}

METRIC_TEXT = (
    ("max-F1", "The highest detection F1 over the mode's threshold grid on each held-out seed; detections on the "
               "reference round match truth puncta within 2 voxels."),
    ("AUPRC", "Average precision over the mode's own grid (W-233 formula)."),
    ("correct-decode fraction", "Accepted reads with the correct gene, divided by the truth amplicons, at the "
                                "threshold selected on the development seeds."),
    ("wrong-gene and false-detection reads", "Accepted reads matched to a truth punctum with the wrong gene, and "
                                             "accepted reads without a truth punctum, at the selected threshold."),
    ("background RMSE and bias", "The estimated background (input minus the background step's output) against the "
                                 "background truth, reference round."),
    ("puncta contrast", "(peak − local background) / noise at each truth punctum on the detection image."),
    ("true-spot intensity CV", "The spread of the median on-channel neighbourhood sums across rounds and channels."),
    ("clipped and saturated fractions", "Voxels with positive input set to 0, and voxels at the dtype maximum, over "
                                        "all rounds."),
    ("colour-call agreement", "Matched reads whose observed colour sequence equals the true codeword."),
    ("per-FOV correct-decode range", "The spread of the correct-decode fraction across a set's three FOVs."),
    ("seed range", "Maximum minus minimum over the held-out seeds 100, 101 and 102; the rule's R_E is the larger "
                   "of the before and after ranges."),
)

COMPARISON_TYPES = (
    ("isolated", "no preprocessing (none) against the method alone."),
    ("ablation", "the full recipe against the recipe without that step (ablation_scalar and ablation_3d name the "
                 "recipe-2 background)."),
    ("sample-level against per-FOV fitting", "each recipe with fit=\"supplied\" (the _sample arms) against "
                                             "fit=\"fov\" on the multi-FOV sets."),
    ("extraction source against detection image", "r2_*_xsrc against r2_*: the same detection, with colour calling "
                                                  "from the bg_corrected snapshot."),
    ("harm test", "histogram matching on clean_unbalanced, reported beside its flag (C7)."),
)


def _quiet(function, *args, **kwargs):
    """Calibrated scenes clip negative noise to 0 (recorded in the manifest); the generator's warning is expected."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"round .* voxel values .* were clipped", RuntimeWarning)
        return function(*args, **kwargs)


class DetectionMismatch(SourceMismatch):
    """A detection on a regenerated image differs from its saved curve row."""


class CalibratedScenes:
    """Regenerate displayed calibrated scenes and outputs; every array must match its manifest checksum."""

    def __init__(self, manifest, workdir):
        self.manifest, self.workdir = manifest, Path(workdir)
        self.runs = {(r.get("condition") or r.get("multi_fov"), r["dtype"], r["seed"], r["arm"], r.get("fov_id")):
                     r["output_sha256"] for r in manifest["runs"]}
        self.records = {(s["condition"], s["dtype"], s["seed"], None): s for s in manifest["scenes"]}
        self.records.update({(s["multi_fov"], s["dtype"], s["seed"], s["fov_id"]): s
                             for s in manifest.get("multi_fov_scenes", [])})
        self.k = {dtype: v["k"] for dtype, v in (manifest.get("saturation_k") or {}).items()}
        self.verified, self._cache = [], {}

    def _record(self, key):
        if key not in self.records:
            raise SourceMismatch(f"the manifest records no scene {key}")
        return self.records[key]

    def _check_scene(self, key, scene):
        if scene.provenance["image_sha256"] != self._record(key)["image_sha256"]:
            raise SourceMismatch(f"regenerated scene {key} differs from the manifest")
        rounds = scene.provenance["image_sha256"]
        self.verified.append(dict(key="/".join(str(k) for k in key if k is not None),
                                  what=f"scene, {len(rounds)} rounds (sha256 of {next(iter(rounds))} shown)",
                                  sha256=rounds[next(iter(rounds))]))

    def _check_output(self, key, images, labels, what):
        digest = _digest(images, labels)
        if digest != self.runs.get(key):
            raise SourceMismatch(f"regenerated {what} {key} differs from the manifest output_sha256")
        self.verified.append(dict(key="/".join(str(k) for k in key if k is not None), what=what, sha256=digest))

    def single(self, condition, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED):
        """Scene, truths and recipes of one single-FOV condition (the scene checked against the manifest)."""
        key = (condition, dtype, seed, None)
        if key not in self._cache:
            record = self._record(key)
            _book, preset = harness.calibrated_scene_preset("clean", dtype)
            shape, count = tuple(record["shape_zyx"]), record["count"]
            book, config = harness.calibrated_scene_config(
                condition, dtype=dtype, seed=seed, k=self.k.get(dtype) if condition == "saturation" else None,
                shape=None if shape == tuple(preset.shape_zyx) else shape,
                count=None if count == preset.count else count)
            scene, background, signal, truth = _quiet(harness.generate, book, config)
            self._check_scene(key, scene)
            radius, _uncapped = harness.calibrated_background_radius(config)
            self._cache[key] = dict(book=book, scene=scene, background=background, signal=signal, truth=truth,
                                    fov_id=config.FOV_id, recipes=harness.arms(radius),
                                    register=condition == "combined_geometry", arms={})
        return self._cache[key]

    def processed(self, condition, arm, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED):
        """The preprocessed FOV of one arm; its output (every round) must equal the manifest's."""
        data = self.single(condition, dtype, seed)
        if arm not in data["arms"]:
            fov = _quiet(harness.preprocess, harness.make_fov(data["scene"], data["book"], data["fov_id"],
                                                              self.workdir), data["recipes"][arm], data["register"])
            self._check_output((condition, dtype, seed, arm, None), fov.images, data["book"].round_labels, "output")
            data["arms"][arm] = fov
        return data["arms"][arm]

    def multi(self, name, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED):
        """The FOVs of one multi-FOV set (each scene checked against the manifest)."""
        key = (name, dtype, seed, "set")
        if key not in self._cache:
            spec, fovs = harness.MULTI_FOV[name], {}
            for k, (role, _n, gain) in enumerate(zip(harness.FOV_ROLES[name], spec["counts"], spec["gains"])):
                fov_id = f"Position{k + 1:03d}"
                record = self._record((name, dtype, seed, fov_id))
                book, config = harness.calibrated_scene_config(
                    spec["base"], dtype=dtype, seed=seed, fov_id=fov_id, gain=gain, count=record["count"],
                    scene_key=f"{harness.CALIBRATED_VERSION}/multi-fov/{fov_id}")
                fovs[fov_id] = (*_quiet(harness.generate, book, config), role, config)
                self._check_scene((name, dtype, seed, fov_id), fovs[fov_id][0])
            radius, _uncapped = harness.calibrated_background_radius(config)
            self._cache[key] = dict(book=book, fovs=fovs, radius=radius, arms={}, fitted=set())
        return self._cache[key]

    def processed_fov(self, name, fov_id, arm, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED):
        """One FOV of a set preprocessed by a multi-FOV arm; supplied arms fit over the whole set first."""
        data = self.multi(name, dtype, seed)
        if (fov_id, arm) not in data["arms"]:
            recipe_name, fit = harness.MULTI_FOV_ARMS[arm]
            path = self.workdir / f"supplied-{name}-{dtype}-{seed}-{arm}.json"
            recipe = harness.arms(data["radius"], fit, path)[recipe_name]
            if fit == "supplied" and arm not in data["fitted"]:
                _quiet(harness.fit_supplied, recipe, data["fovs"], data["book"], self.workdir)
                data["fitted"].add(arm)
            scene = data["fovs"][fov_id][0]
            fov = _quiet(harness.preprocess, harness.make_fov(scene, data["book"], fov_id, self.workdir), recipe,
                         False)
            self._check_output((name, dtype, seed, arm, fov_id), fov.images, data["book"].round_labels, "output")
            data["arms"][(fov_id, arm)] = fov
        return data["arms"][(fov_id, arm)]


def realized_peaks(truth, signal, book):
    """Change 9: each truth punctum's realized peak, the noise-free signal maximum over the 3×3×3 voxels around
    its rounded centre on its reference-round channel, in truth order."""
    peaks = []
    for row in truth.itertuples():
        center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
        volume = signal[..., book.color_to_channel[row.color_sequence[0]]]
        peaks.append(float(volume[harness._box(volume.shape, center, (1, 1, 1))].max()))
    return np.asarray(peaks, dtype=np.float64)


def calibrated_dim_punctum(truth, signal, book):
    """Change 9: the emitting reference-round truth punctum with the lowest realized peak whose centre is in
    bounds (truth holds exactly those puncta, see harness.generate); ties go to the first in truth order."""
    peaks = realized_peaks(truth, signal, book)
    if not len(peaks):
        return dict(center=tuple(n // 2 for n in signal.shape[:3]), channel=0, peak=None, amplicon_id=None,
                    index=None)
    index = int(np.argmin(peaks))
    row = truth.iloc[index]
    return dict(center=tuple(int(round(v)) for v in (row.z, row.y, row.x)),
                channel=book.color_to_channel[row.color_sequence[0]], peak=float(peaks[index]),
                amplicon_id=str(row.amplicon_id), index=index)


def detect(fov, truth, mode, value):
    """Detection, extraction, decoding and filtering at one threshold, matched to truth as the harness's sweep.

    Returns the spots, cutoffs, the truth index of each matched spot, each matched truth punctum's call and
    the counts that the saved curves hold.
    """
    spots, reads, cutoffs = harness._pipeline(fov, value, mode)
    metadata = fov.metadata[fov.rounds.reference_round]
    result = harness.match_points(truth[["z", "y", "x"]].to_numpy(float), spots[["z", "y", "x"]].to_numpy(float),
                                  reference_metadata=metadata, observed_metadata=metadata, **harness.MATCHING)
    matched = {j: i for i, j, _ in result.details["matched_pairs"]}
    correct = wrong = false = 0
    calls = {}
    for j, (gene, accepted, sequence) in enumerate(zip(reads.gene_id, reads.accepted, reads.observed_color_sequence)):
        i = matched.get(j)
        right = i is not None and pd.notna(gene) and str(gene) == str(truth.gene_id.iat[i])
        if i is not None:
            calls[i] = dict(sequence=str(sequence), accepted=bool(accepted), correct=bool(accepted and right))
        if not accepted:
            continue
        if i is None:
            false += 1
        elif right:
            correct += 1
        else:
            wrong += 1
    counts = dict(n_detected=len(spots), n_matched=len(matched), reads_correct=correct, reads_wrong_gene=wrong,
                  reads_false_detection=false)
    return dict(spots=spots, cutoffs=[float(c) for c in cutoffs], matched=matched, calls=calls, counts=counts,
                mode=mode, value=value)


def check_detection(curves, selector, found, what):
    """The saved curve row of a displayed detection must hold the same counts and cutoffs, or raise."""
    rows = curves
    for column, value in selector.items():
        rows = rows[rows[column] == value]
    if len(rows) != 1:
        raise DetectionMismatch(f"{what}: {len(rows)} saved curve rows match {selector}")
    row = rows.iloc[0]
    for column, value in found["counts"].items():
        if int(row[column]) != int(value):
            raise DetectionMismatch(f"{what}: {column} {value} differs from the saved {row[column]}")
    if not np.allclose(json.loads(row["cutoffs"]), found["cutoffs"], rtol=0, atol=1e-9):
        raise DetectionMismatch(f"{what}: cutoffs {found['cutoffs']} differ from the saved {row['cutoffs']}")
    return dict(what=what, **selector, **found["counts"])


def _with_sets(diagnostics):
    """Diagnostics with the multi-FOV columns; a run without multi-FOV sets saves them without."""
    missing = {c: np.nan for c in ("multi_fov", "fov_id") if c not in diagnostics}
    return diagnostics.assign(**missing) if missing else diagnostics


def _scoped(points):
    """Operating points with fov_scope; a run without multi-FOV sets saves single-FOV rows without it."""
    return points if "fov_scope" in points else points.assign(fov_scope="single_fov")


def selected_threshold(points, name, dtype, arm, mode, multi):
    """(value, at_grid_edge) of the development-selected operating point, from operating_points.csv."""
    scope = "multi_fov_pooled" if multi else "single_fov"
    points = _scoped(points)
    rows = points[(points.fov_scope == scope) & (points.condition == name) & (points.dtype == dtype)
                  & (points.arm == arm) & (points.threshold_mode == mode) & (points.operating_point == "dev_max_f1")]
    if len(rows) != 1:
        raise SourceMismatch(f"operating_points.csv has {len(rows)} dev_max_f1 rows for {name} {dtype} {arm} {mode}")
    return float(rows.threshold.iat[0]), _truthy(rows.at_grid_edge.iat[0])


def comparison_spec(method, comparison):
    """(before, after, targeted) of one comparison of the rerun."""
    for m, c, before, after, targeted in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS:
        if (m, c) == (method, comparison):
            return before, after, tuple(targeted)
    raise KeyError((method, comparison))


def _role(method, name):
    _before, _after, targeted = comparison_spec(method, CARDS[method]["comparison"])
    if name in targeted:
        return "targeted"
    if name in ("clean", "mf_density_clean"):
        return "harm_check"
    return dict(harness.calibrated_extra_roles(method, CARDS[method]["comparison"])).get(name, "combined")


def card_rows(method, scenes, sources, checks):
    """The displayed rows of a card: verified images before and after, truths and both modes' detections."""
    card = CARDS[method]
    before, after, _targeted = comparison_spec(method, card["comparison"])
    arms = dict(before=before, after=after)
    points = sources.table("tables/operating_points.csv")
    diagnostics = _with_sets(sources.table("tables/diagnostics.csv"))
    rows = []
    for entry in card["rows"]:
        multi = isinstance(entry, tuple)
        name, fov_role = entry if multi else (entry, None)
        if multi:
            data = scenes.multi(name)
            fov_id = next(f for f, v in data["fovs"].items() if v[4] == fov_role)
            scene, background, signal, truth, _r, _c = data["fovs"][fov_id]
            book = data["book"]
            fovs = {side: scenes.processed_fov(name, fov_id, arm) for side, arm in arms.items()}
            curves = sources.table("curves/multi_fov_curves.csv")
            selector = dict(multi_fov=name, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED, fov_id=fov_id)
            key = dict(condition=name, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED, fov_id=fov_id)
        else:
            data = scenes.single(name)
            scene, background, signal, truth, book = (data["scene"], data["background"], data["signal"],
                                                      data["truth"], data["book"])
            fov_id = None
            fovs = {side: scenes.processed(name, arm) for side, arm in arms.items()}
            curves = sources.table("curves/pr_curves.csv")
            selector = dict(condition=name, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED)
            key = dict(condition=name, dtype=DISPLAY_DTYPE, seed=DISPLAY_SEED)
        ref = book.round_labels[0]
        images = {side: {r: fov.images[r] for r in book.round_labels} for side, fov in fovs.items()}
        if card.get("snapshot"):
            # The xsrc arm extracts from its bg_corrected snapshot, the output of the background step alone.
            snapshot = {r: s[harness.SNAPSHOT] for r, s in fovs["after"].snapshots.items()}
            scenes._check_output((name, DISPLAY_DTYPE, DISPLAY_SEED, "scalar", fov_id), snapshot, book.round_labels,
                                 f"{after} {harness.SNAPSHOT} snapshot (equal to the scalar arm)")
            images["after"] = snapshot
        estimate = None
        if card.get("background"):
            corrected = fovs["after"].snapshots[ref][harness.SNAPSHOT]
            estimate = scene.rounds[ref].astype(np.float64) - corrected.astype(np.float64)
        detections = {}
        for side, arm in arms.items():
            for mode in MODES:
                value, edge = selected_threshold(points, name, DISPLAY_DTYPE, arm, mode, multi)
                found = detect(fovs[side], truth, mode, value)
                checks.append(check_detection(curves, dict(selector, arm=arm, threshold_mode=mode, threshold=value),
                                              found, f"{method}: {name} {side} {arm} {mode}"))
                detections[(side, mode)] = dict(found, edge=edge)
        noise5 = {}
        for side, arm in arms.items():
            record_arm = "scalar" if side == "after" and card.get("snapshot") else arm
            noise5[side] = panel_cutoffs(diagnostics, key, record_arm, book)
            _check_cutoffs(images[side][ref], noise5[side], f"{method} {name} {side}")
        if card.get("snapshot"):
            # Change 6 companion: the xsrc arm's detection image, on which its detection cutoffs apply.
            noise5["after_detection"] = panel_cutoffs(diagnostics, key, after, book)
            _check_cutoffs(fovs["after"].images[ref], noise5["after_detection"], f"{method} {name} after detection")
        rows.append(dict(name=name, role=_role(method, name), fov_id=fov_id, fov_role=fov_role, book=book,
                         truth=truth, background=background[ref], signal=signal[ref], images=images,
                         detection_images={side: fov.images for side, fov in fovs.items()}, arms=arms,
                         detections=detections, noise5=noise5, estimate=estimate,
                         dim=calibrated_dim_punctum(truth, signal[ref], book),
                         peaks=realized_peaks(truth, signal[ref], book)))
    return rows


def _row_title(row):
    where = f", {row['fov_role']} FOV {row['fov_id']}" if row["fov_id"] else ""
    return f"{row['name']} ({ROLE_LABEL[row['role']]}{where})"


def _lead(row, k):
    """Figure title lines naming the row on its first panel (two lines, so long names are not clipped)."""
    where = f", {row['fov_role']} FOV {row['fov_id']}" if row["fov_id"] else ""
    return f"{row['name']}\n({ROLE_LABEL[row['role']]}{where})\n" if k == 0 else "\n\n"


def _crop(image, center, half, whole_rows=False):
    """A NaN-padded crop of a 2D image around center (row, column); whole_rows keeps every row."""
    padded = np.pad(image.astype(np.float64), ((0, 0) if whole_rows else (half, half), (half, half)),
                    constant_values=np.nan)
    r, c = center
    return padded[slice(None) if whole_rows else slice(r, r + 2 * half + 1), c:c + 2 * half + 1]


def _cmap(plt, name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#d9d9e8")
    return cmap


def _small(axis, title, xlabel=None, ylabel=None, size=7.5):
    axis.set_title(title, fontsize=size)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=7)
    axis.tick_params(labelsize=6)


def _marks_in_crop(truth, book, channel, dim, half, view):
    """Truth centres on the channel near the plane, in crop coordinates: view "z" (Y–X) or "xz" (Z–X)."""
    z0, y0, x0 = dim
    marks = []
    for row in truth.itertuples():
        if book.color_to_channel[row.color_sequence[0]] != channel:
            continue
        if view == "z" and abs(row.z - z0) <= 1 and abs(row.y - y0) <= half and abs(row.x - x0) <= half:
            marks.append((row.x - x0 + half, row.y - y0 + half))
        elif view == "xz" and abs(row.y - y0) <= 1 and abs(row.x - x0) <= half:
            marks.append((row.x - x0 + half, row.z))
    return marks


def _is_snapshot(label):
    return label.endswith("snapshot)")


def panels_figure(rows, labels):
    """Changes 1, 2 and 9: per row, the cropped Z slice and XZ view around the dim punctum before and after in
    one shared linear display range, after − before, and the line profile in two adjacent panels, one per
    threshold mode, each with that mode's detection cutoffs."""
    facts = []
    with _pyplot() as plt:
        figure = plt.figure(figsize=(22, 3.5 * len(rows)), layout="constrained")
        grid = figure.add_gridspec(len(rows), 8, width_ratios=(1, 1, 1, 1, 1, 1, 1.7, 1.7))
        for i, row in enumerate(rows):
            book, dim = row["book"], row["dim"]
            (z0, y0, x0), c = dim["center"], dim["channel"]
            ref, channel = book.round_labels[0], book.channel_labels[c]
            volumes = {side: row["images"][side][ref][..., c].astype(np.float64) for side in ("before", "after")}
            low = min(float(v.min()) for v in volumes.values())
            high = max(float(np.percentile(v, 99.9)) for v in volumes.values())
            high = high if high > low else low + 1.0
            difference = volumes["after"] - volumes["before"]
            span = max(float(np.percentile(np.abs(difference), 99.9)), 1e-9)
            views = [(side, _crop(volumes[side][z0], (y0, x0), CROP_YX), "z") for side in ("before", "after")]
            views.append(("after − before", _crop(difference[z0], (y0, x0), CROP_YX), "z"))
            views += [(side, _crop(volumes[side][:, y0, :], (0, x0), CROP_YX, True), "xz")
                      for side in ("before", "after")]
            views.append(("after − before", _crop(difference[:, y0, :], (0, x0), CROP_YX, True), "xz"))
            for k, (side, image, view) in enumerate(views):
                axis = figure.add_subplot(grid[i, k])
                diff = side == "after − before"
                axis.imshow(image, cmap=_cmap(plt, "RdBu_r" if diff else "gray"), vmin=-span if diff else low,
                            vmax=span if diff else high, interpolation="nearest", origin="upper",
                            aspect=XZ_STRETCH if view == "xz" else 1)
                marks = _marks_in_crop(row["truth"], book, c, (z0, y0, x0), CROP_YX, view)
                if marks:
                    xs, ys = zip(*marks)
                    axis.scatter(xs, ys, s=70, facecolors="none", edgecolors="cyan", linewidths=1)
                axis.scatter([CROP_YX], [CROP_YX if view == "z" else z0], s=130, marker="s", facecolors="none",
                             edgecolors="red", linewidths=1.5)
                arm = "" if diff else f": {labels[side]}"
                plane = f"z={z0}" if view == "z" else f"XZ y={y0}, Z ×{XZ_STRETCH}"
                shown = f"±{span:.3g}" if diff else f"shared [{low:.4g}, {high:.4g}]"
                title = _lead(row, k) + f"{side}{arm}\n{plane}, {channel}; {shown}"
                _small(axis, title, "x (crop)", "y (crop)" if view == "z" else "z")
            for k, mode in enumerate(MODES):
                axis = figure.add_subplot(grid[i, 6 + k])
                for side, colour in (("before", "0.35"), ("after", "tab:orange")):
                    axis.plot(row["images"][side][ref][z0, y0, :, c], color=colour, marker=".", markersize=3,
                              label=f"{side}: {labels[side]}")
                    if _is_snapshot(labels[side]):
                        continue
                    found = row["detections"][(side, mode)]
                    axis.axhline(found["cutoffs"][c], color=colour, linewidth=1.2, linestyle="--",
                                 label=f"{side} detection cutoff at {found['value']:g}")
                axis.axvline(x0, color="red", linestyle=":", linewidth=1)
                _small(axis, f"{mode.upper()} MODE: {mode} detection cutoffs\nprofile along X at z={z0}, y={y0}, "
                             f"{channel}\ndim punctum {dim['amplicon_id']}, realized peak {fmt(dim['peak'])}",
                       "x (voxel)", "intensity")
                axis.legend(fontsize=6)
            facts.append(dict(row=_row_title(row), z=z0, y=y0, x=x0, channel=channel, range=(low, high),
                              difference=span, dim=dim["amplicon_id"], peak=dim["peak"]))
        png = _png(figure)
    return png, facts


def _projection(images, book):
    """Maximum over Z and channels of the reference round."""
    return images[book.round_labels[0]].astype(np.float64).max(axis=(0, 3))


def overlay_figure(rows):
    """Change 3: true positives, false positives and misses at each mode's development-selected threshold."""
    with _pyplot() as plt:
        figure, axes = plt.subplots(len(rows), 4, figsize=(15, 4.2 * len(rows)), layout="constrained",
                                    squeeze=False)
        for i, row in enumerate(rows):
            book, truth = row["book"], row["truth"]
            projections = {side: _projection(row["detection_images"][side], book) for side in ("before", "after")}
            low = min(float(p.min()) for p in projections.values())
            high = max(float(np.percentile(p, 99.5)) for p in projections.values())
            for k, (mode, side) in enumerate((m, s) for m in MODES for s in ("before", "after")):
                axis = axes[i, k]
                found = row["detections"][(side, mode)]
                axis.imshow(projections[side], cmap="gray", vmin=low, vmax=high if high > low else low + 1,
                            interpolation="nearest")
                spots = found["spots"]
                tp = [j for j in range(len(spots)) if j in found["matched"]]
                fp = [j for j in range(len(spots)) if j not in found["matched"]]
                missed = sorted(set(range(len(truth))) - set(found["matched"].values()))
                axis.scatter(spots.x.iloc[tp], spots.y.iloc[tp], s=45, facecolors="none", edgecolors="lime",
                             linewidths=1.1, label=f"true positive ({len(tp)})")
                axis.scatter(spots.x.iloc[fp], spots.y.iloc[fp], s=40, marker="x", color="red", linewidths=1.1,
                             label=f"false positive ({len(fp)})")
                axis.scatter(truth.x.iloc[missed], truth.y.iloc[missed], s=55, marker="s", facecolors="none",
                             edgecolors="yellow", linewidths=1.1, label=f"missed ({len(missed)})")
                edge = ", grid edge" if found["edge"] else ""
                title = _lead(row, k) + (
                    f"{mode} mode: {side} {row['arms'][side]}\nthreshold_value {found['value']:g}{edge}")
                _small(axis, title, "x (voxel)", "y (voxel)")
                axis.legend(fontsize=6, loc="lower right", framealpha=0.7)
        png = _png(figure)
    return png


def _neighbourhood(images, book, center):
    """Channel × round neighbourhood sums (the harness's extraction box) at a reference-round centre."""
    radius = harness.EXTRACTION.neighborhood_radius_zyx
    values = np.zeros((len(book.channel_labels), len(book.round_labels)))
    for r, name in enumerate(book.round_labels):
        volume = images[name]
        values[:, r] = volume[harness._box(volume.shape[:3], center, radius)].astype(np.float64).sum(axis=(0, 1, 2))
    return values


def colour_puncta(peaks, count=4):
    """Truth indices shown in the colour-calling view: the dim punctum, then the puncta at the 25th, 50th and
    90th percentiles of the realized peak."""
    if not len(peaks):
        return []
    order = np.argsort(peaks, kind="stable")
    chosen = [int(order[0])]
    for q in (0.25, 0.5, 0.9):
        index = int(order[int(round(q * (len(order) - 1)))])
        if index not in chosen:
            chosen.append(index)
    return chosen[:count]


def _call_text(call):
    if call is None:
        return "not detected"
    state = "correct" if call["correct"] else "accepted, wrong gene" if call["accepted"] else "rejected"
    return f"{call['sequence']} ({state})"


def colour_figure(row, labels):
    """Change 4: channel × round intensity vectors before and after at a few truth puncta, with the true colour
    sequence boxed in red (drawn once: they do not depend on the threshold mode), and each punctum's calls in two
    adjacent panels, one per threshold mode."""
    chosen = colour_puncta(row["peaks"])
    book, truth = row["book"], row["truth"]
    with _pyplot() as plt:
        figure, axes = plt.subplots(max(1, len(chosen)), 4, figsize=(17, 3.3 * max(1, len(chosen))),
                                    layout="constrained", squeeze=False, width_ratios=(1.25, 1.25, 1, 1))
        for k, index in enumerate(chosen):
            record = truth.iloc[index]
            center = tuple(int(round(v)) for v in (record.z, record.y, record.x))
            sequence = str(record.color_sequence)
            dim = " (dim punctum)" if index == row["dim"]["index"] else ""
            for s, side in enumerate(("before", "after")):
                axis = axes[k, s]
                values = _neighbourhood(row["images"][side], book, center)
                shown = axis.imshow(values, cmap="viridis", aspect="auto")
                figure.colorbar(shown, ax=axis, shrink=0.8).ax.tick_params(labelsize=6)
                for r, colour in enumerate(sequence):
                    axis.add_patch(plt.Rectangle((r - 0.5, book.color_to_channel[colour] - 0.5), 1, 1, fill=False,
                                                 edgecolor="red", linewidth=2.2))
                for (c, r), value in np.ndenumerate(values):
                    axis.text(r, c, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color="white")
                axis.set_xticks(range(len(book.round_labels)), book.round_labels)
                axis.set_yticks(range(len(book.channel_labels)), book.channel_labels)
                _small(axis, f"{record.amplicon_id}{dim}, true {sequence}, realized peak {row['peaks'][index]:.3g}\n"
                             f"{side}: {labels[side]} (same in both modes)")
            for m, mode in enumerate(MODES):
                axis = axes[k, 2 + m]
                axis.axis("off")
                lines = [f"{mode.upper()} MODE: calls of {record.amplicon_id}", f"true sequence {sequence}", ""]
                for side in ("before", "after"):
                    found = row["detections"][(side, mode)]
                    lines.append(f"{side} ({row['arms'][side]}, value {found['value']:g}):")
                    lines.append(f"  {_call_text(found['calls'].get(index))}")
                axis.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace",
                          transform=axis.transAxes,
                          bbox=dict(boxstyle="round", facecolor="#dfe8f8" if mode == "noise" else "#ece2f5",
                                    edgecolor="0.6"))
        png = _png(figure)
    return png, [str(truth.amplicon_id.iat[i]) for i in chosen]


def background_figure(rows, label):
    """Change 5: the estimated against the true background on the dim punctum's slice and channel, with the
    residual."""
    with _pyplot() as plt:
        figure = plt.figure(figsize=(16, 3.7 * len(rows)), layout="constrained")
        grid = figure.add_gridspec(len(rows), 4, width_ratios=(1, 1, 1, 1.7))
        for i, row in enumerate(rows):
            (z0, y0, x0), c = row["dim"]["center"], row["dim"]["channel"]
            channel = row["book"].channel_labels[c]
            truth, estimate = row["background"][..., c], row["estimate"][..., c]
            residual = estimate - truth
            low = min(float(truth[z0].min()), float(estimate[z0].min()))
            high = max(float(truth[z0].max()), float(estimate[z0].max()))
            high = high if high > low else low + 1.0
            span = max(float(np.abs(residual[z0]).max()), 1e-9)
            panels = (("background truth", truth[z0], False), (f"estimate: input − {label} bg_corrected",
                                                               estimate[z0], False),
                      ("residual: estimate − truth", residual[z0], True))
            for k, (title, image, diverging) in enumerate(panels):
                axis = figure.add_subplot(grid[i, k])
                shown = axis.imshow(image, cmap="RdBu_r" if diverging else "magma", vmin=-span if diverging else low,
                                    vmax=span if diverging else high, interpolation="nearest")
                figure.colorbar(shown, ax=axis, shrink=0.8).ax.tick_params(labelsize=6)
                shown_range = f"±{span:.3g}" if diverging else f"shared [{low:.3g}, {high:.3g}]"
                _small(axis, _lead(row, k) + f"{title}\nz={z0}, {channel}; "
                       f"{shown_range}", "x (voxel)", "y (voxel)")
            axis = figure.add_subplot(grid[i, 3])
            axis.plot(truth[z0, y0, :], color="tab:blue", label="background truth")
            axis.plot(estimate[z0, y0, :], color="tab:orange", label="estimate")
            axis.plot(residual[z0, y0, :], color="tab:red", linestyle=":", label="residual")
            axis.axvline(x0, color="red", linestyle=":", linewidth=1)
            axis.legend(fontsize=6)
            _small(axis, f"Profile along X at z={z0}, y={y0}, {channel} (input units)", "x (voxel)", "intensity")
        png = _png(figure)
    return png


def signal_channels(truth, book):
    """Change 6: the channels with at least one truth punctum in the reference round."""
    return sorted({book.color_to_channel[str(s)[0]] for s in truth.color_sequence})


def _histogram_entries(row, labels):
    """(label, image, noise cutoffs, detections or None) per histogram row. A snapshot is not detected on, so
    its row keeps the noise cutoff only, and a companion row shows the detection image with the cutoffs."""
    ref, entries = row["book"].round_labels[0], []
    for side in ("before", "after"):
        if _is_snapshot(labels[side]):
            entries.append((f"{side}: {labels[side]}, extraction image", row["images"][side][ref],
                            row["noise5"][side], None))
            entries.append((f"{side}: {row['arms'][side]}\ndetection image", row["detection_images"][side][ref],
                            row["noise5"]["after_detection"], {m: row["detections"][(side, m)] for m in MODES}))
        else:
            entries.append((f"{side}: {labels[side]}", row["images"][side][ref], row["noise5"][side],
                            {m: row["detections"][(side, m)] for m in MODES}))
    return entries


def histogram_figure(rows, labels):
    """Change 6: histograms of the signal channels only. Each channel has two adjacent panels, noise mode and
    adaptive mode, each with the noise cutoff (value 5) and that mode's detection cutoff at its
    development-selected value; an image without detection (a snapshot) is drawn once, with the noise cutoff."""
    width = max(1, max(len(signal_channels(r["truth"], r["book"])) for r in rows))
    entries = [(row, entry) for row in rows for entry in _histogram_entries(row, labels)]
    with _pyplot() as plt:
        figure = plt.figure(figsize=(2.4 * 2 * width, 2.35 * len(entries)), layout="constrained")
        grid = figure.add_gridspec(len(entries), 2 * width)
        previous = None
        for i, (row, (label, image, noise5, detections)) in enumerate(entries):
            book = row["book"]
            channels = signal_channels(row["truth"], book)
            first = row is not previous
            previous = row
            for k in range(width):
                if k >= len(channels):
                    figure.add_subplot(grid[i, 2 * k:2 * k + 2]).axis("off")
                    continue
                c = channels[k]
                bins = np.arange(0, 257) if image.dtype == np.uint8 else 128
                slots = [(None, grid[i, 2 * k:2 * k + 2])] if detections is None else [
                    (mode, grid[i, 2 * k + m]) for m, mode in enumerate(MODES)]
                for mode, slot in slots:
                    axis = figure.add_subplot(slot)
                    axis.hist(image[..., c].ravel(), bins=bins, histtype="stepfilled", color="0.75", log=True)
                    axis.axvline(noise5[c], color="0.15", linestyle="--", linewidth=1.2,
                                 label=f"noise cutoff (value 5): {noise5[c]:.4g}")
                    if mode is not None:
                        found = detections[mode]
                        axis.axvline(found["cutoffs"][c], linewidth=1.5, color=MODE_STYLE[mode]["color"],
                                     label=f"detection cutoff at {found['value']:g}: {found['cutoffs'][c]:.4g}")
                    head = _lead(row, 0) if first and k == 0 and mode != MODES[-1] else "\n\n"
                    where = (f"{mode.upper()} MODE" if mode else "not detected on (same in both modes)")
                    _small(axis, head + f"{label}\n{book.channel_labels[c]}: {where}", "intensity", "voxels (log)",
                           size=7)
                    axis.legend(fontsize=5.5)
        png = _png(figure)
    return png


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truthy(value):
    return value is True or (isinstance(value, np.bool_) and bool(value)) or value == "True"


def _card_records(method, comparisons):
    """Comparison rows of a method, ordered by comparison, role, dtype and mode."""
    part = comparisons[comparisons.method == method].copy()
    order = [c for m, c, *_ in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS if m == method]
    part = part.assign(_c=part.comparison.map(order.index), _r=part.role.map(ROLE_ORDER.index),
                       _d=part.dtype.map(harness.DTYPES.index), _m=part.threshold_mode.map(MODES.index))
    return part.sort_values(["_c", "_r", "_d", "_m"], kind="stable")


def dot_figure(method, comparisons):
    """Change 7: Δ max-F1 and Δ correct-decode per comparison row with seed-range bars and the provisional flag
    band ±max(R_E, 0.02), both dtypes and both modes side by side."""
    part = _card_records(method, comparisons)
    keys = list(dict.fromkeys(zip(part.comparison, part.condition, part.role, part.dtype)))
    markers = dict(targeted="o", harm_check="s", combined="D", reported="^", harm_test="v")
    colours = dict(targeted="tab:blue", harm_check="tab:green", combined="0.4", reported="tab:olive",
                   harm_test="tab:red")
    with _pyplot() as plt:
        figure, axes = plt.subplots(1, 4, figsize=(17, 2.2 + 0.27 * len(keys)), layout="constrained", sharey=True)
        for k, (mode, endpoint) in enumerate((m, e) for m in MODES for e in harness.RULE_ENDPOINTS):
            axis = axes[k]
            rows = {(r["comparison"], r["condition"], r["role"], r["dtype"]): r
                    for r in _records(part[part.threshold_mode == mode])}
            for y, key in enumerate(keys):
                record = rows.get(key)
                if record is None:
                    continue
                span = _float(record.get(f"rule_range_{endpoint}"))
                if np.isfinite(span):
                    band = max(span, harness.LOW_BENEFIT_POINTS)
                    axis.barh(y, 2 * band, left=-band, height=0.8, color="#e4e4ee", zorder=0)
                mean = _float(record.get(f"{endpoint}_delta_mean"))
                if not np.isfinite(mean):
                    axis.text(0, y, "undefined", fontsize=6, va="center", ha="center", color="red")
                    continue
                low, high = _float(record.get(f"{endpoint}_delta_min")), _float(record.get(f"{endpoint}_delta_max"))
                gap = key[2] == "targeted" and not _truthy(record.get("eligible"))
                axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=markers[key[2]], color=colours[key[2]],
                              markerfacecolor="white" if gap else colours[key[2]], markersize=5, capsize=2,
                              linewidth=1)
            axis.axvline(0, color="black", linewidth=0.8)
            _small(axis, f"{mode} mode: Δ {ENDPOINT_LABEL[endpoint]}\n(after − before; held-out seeds)", "Δ")
        axes[0].set_yticks(range(len(keys)), [f"{c} · {n} ({ROLE_LABEL[r]}) · {d}" for c, n, r, d in keys],
                           fontsize=6.5)
        axes[0].invert_yaxis()
        handles = [plt.Line2D([], [], marker=markers[r], color=colours[r], linestyle="", label=ROLE_LABEL[r])
                   for r in ROLE_ORDER if any(key[2] == r for key in keys)]
        handles.append(plt.Line2D([], [], marker="o", color="tab:blue", markerfacecolor="white", linestyle="",
                                  label="targeted, fixture gap (not in T*)"))
        handles.append(plt.Rectangle((0, 0), 1, 1, color="#e4e4ee", label="flag band ±max(R_E, 0.02), provisional"))
        handles.append(plt.Line2D([], [], color="black", marker="|", linestyle="-",
                                  label="bar: min to max of the per-seed Δ"))
        figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=7)
        png = _png(figure)
    return png, len(keys)


def pr_points(curves, name, dtype, arm, mode, name_column="condition"):
    """Mean precision and recall over the held-out seeds at each grid value, in threshold order."""
    part = curves[(curves[name_column] == name) & (curves.dtype == dtype) & (curves.arm == arm)
                  & (curves.threshold_mode == mode) & curves.seed.isin(harness.EVAL_SEEDS)]
    return part.groupby("threshold")[["precision", "recall"]].mean().reset_index().sort_values("threshold")


def collapsed(points):
    """True when every grid value gives the same mean precision and recall: a curve collapsed to one point."""
    finite = points.dropna(subset=["precision", "recall"])
    return len(finite) > 1 and len({(round(p, 12), round(r, 12)) for p, r in zip(finite.precision,
                                                                                    finite.recall)}) == 1


def pr_figure(method, comparison, curves, mad_zero, name_column="condition"):
    """Change 8: small multiples of the method's before/after pair only, one panel per condition, dtype and mode.
    Curves that collapse to a point are marked, not dropped, and named MAD 0 where the manifest records it."""
    before, after, targeted = comparison_spec(method, comparison)
    harm = "mf_density_clean" if method == "sample_level_fitting" else "clean"
    roles = [(c, "targeted") for c in targeted] + [(harm, "harm_check")]
    roles += [] if method == "sample_level_fitting" else [("combined", "combined")]
    roles += list(harness.calibrated_extra_roles(method, comparison))
    panels = [(c, r, d) for c, r in roles for d in harness.DTYPES
              if ((curves[name_column] == c) & (curves.dtype == d)).any()]
    notes = []
    with _pyplot() as plt:
        figure, axes = plt.subplots(len(panels), 2, figsize=(9.5, 2.9 * len(panels)), layout="constrained",
                                    squeeze=False)
        for i, (condition, role, dtype) in enumerate(panels):
            for k, mode in enumerate(MODES):
                axis = axes[i, k]
                for arm, colour in ((before, "0.35"), (after, "tab:orange")):
                    points = pr_points(curves, condition, dtype, arm, mode, name_column).dropna(
                        subset=["precision", "recall"])
                    if collapsed(points):
                        reason = " (MAD 0)" if (condition, dtype, arm) in mad_zero and mode == "noise" else ""
                        axis.scatter(points.recall.iloc[:1], points.precision.iloc[:1], s=170, marker="*",
                                     color=colour, edgecolors="black", zorder=3,
                                     label=f"{arm}: every value gives one point{reason}")
                        notes.append(f"{condition} {dtype} {mode} {arm}{reason}")
                        continue
                    axis.plot(points.recall, points.precision, marker="o", markersize=3, color=colour, label=arm)
                    for end in (0, -1):
                        if len(points):
                            axis.annotate(f"{points.threshold.iat[end]:g}",
                                          (points.recall.iat[end], points.precision.iat[end]), fontsize=6,
                                          color=colour)
                axis.set_xlim(0, 1.02)
                axis.set_ylim(0, 1.02)
                axis.legend(fontsize=6, loc="lower left")
                _small(axis, f"{condition} ({ROLE_LABEL[role]}), {dtype}, {mode} mode", "recall", "precision")
        png = _png(figure)
    return png, notes


# --- Calibrated report: HTML ------------------------------------------------------------------------

CALIBRATED_CSS = CSS + """
.banner { background: #fff3cd; border: 1px solid #e0a800; padding: 8px 12px; }
.card { border: 1px solid #c9ccd6; border-radius: 6px; padding: 2px 14px 10px; margin: 20px 0; background: #fcfcfe; }
.scroll { overflow-x: auto; max-width: 100%; }
.res { padding: 0 4px; border-radius: 3px; white-space: nowrap; font-weight: 600; }
.res-low_benefit { background: #fde2e2; } .res-not_assessable { background: #e6e6e6; }
.res-not_flagged { background: #dff3e3; }
.fail { color: #b00020; font-weight: 600; } .pass { color: #1b6e2a; }
.label { font-size: 10.5px; font-weight: 700; padding: 0 5px; border-radius: 3px; text-transform: uppercase; }
.label.finding { background: #dfe9fb; color: #1a3d7c; } .label.hypothesis { background: #f6e3fb; color: #6b1a7c; }
th.mode-noise { background: #dfe8f8; } th.mode-adaptive { background: #ece2f5; }
table.full { font-size: 9.5px; line-height: 1.25; table-layout: fixed; }
table.full th, table.full td { white-space: normal; overflow-wrap: anywhere; padding: 1px 4px; }
table.full th { word-break: break-all; }
table.fit { width: 100%; table-layout: fixed; }
table.fit th, table.fit td { overflow-wrap: anywhere; }
td code, th code { white-space: normal; word-break: normal; overflow-wrap: anywhere; } span.range { white-space: normal; }
details { margin: 4px 0; } details summary { cursor: pointer; font-size: 12px; }
nav.contents { font-size: 13px; } nav.contents a { margin-right: 12px; }
ul.caveats li { margin-bottom: 3px; }
"""


def _modes_head(key_headers, mode_headers):
    """A two-row header: key columns, then each mode's columns side by side."""
    top = "".join(f"<th rowspan=\"2\">{h}</th>" for h in key_headers)
    top += "".join(f"<th class=\"mode-{m}\" colspan=\"{len(mode_headers)}\">{m} mode</th>" for m in MODES)
    second = "".join(f"<th class=\"mode-{m}\">{h}</th>" for m in MODES for h in mode_headers)
    return f"<thead><tr>{top}</tr><tr>{second}</tr></thead>"


#: Key column widths (px) of the card tables: comparison, condition (role), dtype; wide enough that
#: identifiers wrap only at word breaks.
CARD_KEY_WIDTHS = (135, 130, 72)


def _colgroup(widths):
    """Column widths in px (None: share the rest) for a fitted table."""
    return "<colgroup>" + "".join(f"<col style=\"width:{w}px\">" if w else "<col>" for w in widths) + "</colgroup>"


def mode_table(keys, by_mode, key_columns, mode_columns, source, sources, note="", key_widths=None):
    """Rows keyed by keys; key_columns: (header, f(key)); mode_columns: (header, f(record)) per mode.

    by_mode maps (mode, key) -> record. The modes are shown side by side and never pooled. The table fits the
    page width: key_widths (px) fix the key columns and the mode columns share the rest.
    """
    widths = list(key_widths or [None] * len(key_columns)) + [None] * (len(mode_columns) * len(MODES))
    head = _colgroup(widths) + _modes_head([h for h, _f in key_columns], [h for h, _f in mode_columns])
    body = []
    for key in keys:
        cells = [f"<td>{f(key)}</td>" for _h, f in key_columns]
        for mode in MODES:
            record = by_mode.get((mode, key))
            cells += [f"<td class=\"num\">{'–' if record is None else f(record)}</td>" for _h, f in mode_columns]
        body.append("<tr>" + "".join(cells) + "</tr>")
    names = source if isinstance(source, (list, tuple)) else [source]
    caption = "Source: " + ", ".join(f"{code(s)} (sha256 {code(sources.records[s]['sha256'][:16])}…)" for s in names)
    return (f"<div class=\"scroll\"><table class=\"fit\">{head}<tbody>{''.join(body)}</tbody></table></div>"
            f"<p class=\"caption\">{caption}. {note}</p>")


def _delta(record, stem, digits=3):
    mean, low, high = (record.get(f"{stem}_delta_{s}") for s in ("mean", "min", "max"))
    if mean is None:
        return "–"
    return f"Δ {fmt(mean, digits)} <span class=\"range\">[{fmt(low, digits)}, {fmt(high, digits)}]</span>"


def _ba(record, stem, digits=3, rule=False):
    """before → after means, then Δ mean [min, max] over the held-out seeds (and the rule's R_E)."""
    before, after = record.get(f"{stem}_before_mean"), record.get(f"{stem}_after_mean")
    if before is None and after is None:
        return "–"
    text = f"{fmt(before, digits)} → {fmt(after, digits)}<br>{_delta(record, stem, digits)}"
    if rule and record.get(f"rule_range_{stem}") is not None:
        text += f"<br><span class=\"range\">R_E {fmt(record.get(f'rule_range_{stem}'), 3)}</span>"
    return text


def _direct_cell(record, stem):
    before = spread(record, f"{stem}_before")
    after = spread(record, f"{stem}_after")
    return "–" if before == "–" and after == "–" else f"{before}<br>→ {after}"


def _badge(result):
    return f"<span class=\"res res-{esc(result)}\">{esc(str(result).replace('_', ' '))}</span>"


def _precondition_cell(record):
    if record.get("role") != "targeted":
        return "not applied"
    status = esc(record.get("precondition") or "missing")
    if _truthy(record.get("eligible")):
        return f"<span class=\"pass\">eligible</span><br><span class=\"range\">{status}</span>"
    return f"<span class=\"fail\">fixture gap</span><br><span class=\"range\">{status}</span>"


class Lookups:
    """Operating points and flags indexed for the cards."""

    def __init__(self, sources):
        points = _scoped(sources.table("tables/operating_points.csv"))
        dev = points[points.operating_point == "dev_max_f1"]
        self.selected = {(r["fov_scope"], r["condition"], r["dtype"], r["arm"], r["threshold_mode"]):
                         (r["threshold"], _truthy(r["at_grid_edge"])) for r in _records(dev)}
        self.flags = sources.table("tables/low_benefit_flags.csv")

    def threshold(self, condition, dtype, arm, mode):
        scope = "multi_fov_pooled" if condition in harness.MULTI_FOV else "single_fov"
        value = self.selected.get((scope, condition, dtype, arm, mode))
        if value is None:
            return "–"
        return f"{fmt(value[0])}{' <span class=fail>(grid edge)</span>' if value[1] else ''}"


def card_numbers(method, sources, lookups):
    """Key before/after numbers of a card: downstream metrics per mode side by side, and direct metrics."""
    card = CARDS[method]
    part = _card_records(method, sources.table("tables/comparisons.csv"))
    records = _records(part)
    key_of = lambda r: (r["comparison"], r["before"], r["after"], r["condition"], r["role"], r["dtype"])  # noqa: E731
    keys = list(dict.fromkeys(key_of(r) for r in records))
    by_mode = {(r["threshold_mode"], key_of(r)): r for r in records}
    key_columns = [("comparison", lambda k: f"{code(k[0])}<br>{code(k[1])} → {code(k[2])}"),
                   ("condition (role)", lambda k: f"{code(k[3])}<br>{esc(ROLE_LABEL[k[4]])}"),
                   ("dtype", lambda k: code(k[5]))]
    mode_columns = [("precondition; selected value before / after", lambda r: (
                        f"{_precondition_cell(r)}<br><span class=\"range\">value "
                        f"{lookups.threshold(r['condition'], r['dtype'], r['before'], r['threshold_mode'])} / "
                        f"{lookups.threshold(r['condition'], r['dtype'], r['after'], r['threshold_mode'])}</span>")),
                    ("max-F1", lambda r: _ba(r, "max_f1", rule=True)),
                    ("correct-decode fraction", lambda r: _ba(r, "correct_fraction", rule=True)),
                    ("AUPRC", lambda r: _ba(r, "auprc")),
                    ("reads: wrong-gene; false-detection", lambda r: (
                        f"{_ba(r, 'reads_wrong_gene')}<br>; {_ba(r, 'reads_false_detection')}"))]
    mode_columns += [(DIRECT_LABEL[stem], (lambda s: lambda r: _ba(r, s))(stem)) for stem in card["mode_direct"]]
    downstream = mode_table(keys, by_mode, key_columns, mode_columns, ["tables/comparisons.csv",
                                                                       "tables/operating_points.csv"], sources,
                            "Means over held-out seeds 100–102, before → after; Δ is the mean paired difference with "
                            "[min, max] of the per-seed Δ; R_E is the flag rule's seed range. Max-F1 and AUPRC use "
                            "each mode's grid; reads and the correct-decode fraction use the value selected on the "
                            "development seeds (first column of each mode).", key_widths=CARD_KEY_WIDTHS)
    direct = ""
    stems = [s for s in card["direct"] if f"{s}_before_mean" in part]
    if stems:
        # Direct metrics come from the images alone; they must agree between the two mode rows.
        same = all(by_mode.get(("noise", k), {}).get(f"{s}_{side}_{stat}") ==
                   by_mode.get(("adaptive", k), {}).get(f"{s}_{side}_{stat}")
                   for k in keys for s in stems for side in ("before", "after") for stat in ("mean", "min", "max"))
        if same:
            rows = "".join("<tr>" + "".join(f"<td>{f(k)}</td>" for _h, f in key_columns)
                           + "".join(f"<td class=\"num\">{_direct_cell(by_mode[('noise', k)], s)}</td>"
                                     for s in stems) + "</tr>" for k in keys if ("noise", k) in by_mode)
            head = "".join(f"<th>{h}</th>" for h, _f in key_columns) + "".join(
                f"<th>{DIRECT_LABEL[s]}</th>" for s in stems)
            direct = (f"<h4 id=\"{method}-direct\">Direct metrics (identical in both threshold modes, checked)</h4>"
                      f"<div class=\"scroll\"><table class=\"fit\">"
                      f"{_colgroup(list(CARD_KEY_WIDTHS) + [None] * len(stems))}<thead><tr>{head}</tr></thead>"
                      f"<tbody>{rows}</tbody></table></div><p class=\"caption\">"
                      f"Source: {code('tables/comparisons.csv')}. Before [min, max] → after [min, max] over the "
                      "held-out seeds; they do not depend on the threshold mode because detection is not involved."
                      "</p>")
        else:
            direct = f"<h4 id=\"{method}-direct\">Direct metrics, per mode</h4>" + mode_table(
                keys, by_mode, key_columns, [(DIRECT_LABEL[s], (lambda s: lambda r: _direct_cell(r, s))(s))
                                             for s in stems], "tables/comparisons.csv", sources,
                key_widths=CARD_KEY_WIDTHS)
    return downstream, direct


def card_flags(method, lookups, sources):
    """The flag verdict per comparison, dtype and mode, with the failing clauses."""
    flags = lookups.flags[lookups.flags.method == method]
    order = [c for m, c, *_ in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS if m == method]
    keys = sorted({(r["comparison"], r["dtype"]) for r in _records(flags)},
                  key=lambda k: (order.index(k[0]), harness.DTYPES.index(k[1])))
    by_mode = {}
    for (comparison, dtype, mode), group in flags.groupby(["comparison", "dtype", "threshold_mode"], sort=False):
        by_mode[(mode, (comparison, dtype))] = _records(group)

    def verdict(rows):
        method_row = next(r for r in rows if r["level"] == "method")
        clauses = []
        for r in rows:
            if r["level"] == "targeted_condition":
                state = ("benefit" if _truthy(r.get("benefit")) else esc(r.get("clause") or "no benefit"))
                clauses.append(f"{code(r['condition'])}: {state}")
            elif r["level"] == "clean":
                clauses.append(f"{code(r['condition'] or 'clean')}: "
                               + ("holds" if _truthy(r.get("clean_holds")) else esc(r.get("clause"))))
        anomalies = f"<br><span class=\"fail\">anomaly: {esc(method_row['anomalies'])}</span>" if method_row.get(
            "anomalies") else ""
        return (f"{_badge(method_row['result'])} <span class=\"range\">eligible targets "
                f"{fmt(method_row.get('eligible_targets'))}</span><br>{esc(method_row.get('reason'))}"
                f"<ul style=\"margin:2px 0 0 14px;padding:0\">{''.join(f'<li>{c}</li>' for c in clauses)}</ul>"
                f"{anomalies}")

    head = _colgroup([140, 72, None, None]) + _modes_head(["comparison", "dtype"],
                                                          ["flag verdict (provisional 0.02) and clauses"])
    body = "".join(f"<tr><td>{code(k[0])}</td><td>{code(k[1])}</td>" + "".join(
        f"<td>{verdict(by_mode[(m, k)]) if (m, k) in by_mode else '–'}</td>" for m in MODES)
        + "</tr>" for k in keys)
    digest = sources.records["tables/low_benefit_flags.csv"]["sha256"][:16]
    return (f"<div class=\"scroll\"><table class=\"fit\">{head}<tbody>{body}</tbody></table></div>"
            f"<p class=\"caption\">Source: {code('tables/low_benefit_flags.csv')} (sha256 {code(digest)}…). "
            "Revised rule of item 4, provisional: "
            "not flagged when an eligible targeted condition shows a benefit and clean holds; not assessable when "
            "no targeted condition is eligible (T* empty); low benefit otherwise. The flag informs review and removes "
            "no method.</p>")


def card_harm(sources):
    """Histogram matching's unbalanced-codebook harm test, beside its flag and the balanced clean deltas (C7)."""
    harm = sources.table("tables/harm_test.csv")
    if not len(harm):
        return "<p>This evaluation has no harm-test rows.</p>"
    records = _records(harm)
    keys = list(dict.fromkeys((r["comparison"], r["before"], r["after"], r["dtype"]) for r in records))
    by_mode = {(r["threshold_mode"], (r["comparison"], r["before"], r["after"], r["dtype"])): r for r in records}

    def pair(r, prefix, endpoint):
        return (f"{fmt(r.get(f'{prefix}_delta_{endpoint}'))} "
                f"<span class=\"range\">(harm below −R_E = {fmt(-_float(r.get(f'{prefix}_range_{endpoint}')))})</span>")

    columns = [("harm on clean_unbalanced", lambda r: ("<span class=\"fail\">harm</span>" if _truthy(r["harm"])
                                                       else "<span class=\"pass\">no harm</span>")
                + (f" ({esc(r['harm_endpoints'])})" if r.get("harm_endpoints") else "")),
               ("unbalanced Δ max-F1", lambda r: pair(r, "unbalanced", "max_f1")),
               ("unbalanced Δ correct-decode", lambda r: pair(r, "unbalanced", "correct_fraction")),
               ("balanced clean Δ max-F1", lambda r: pair(r, "clean", "max_f1")),
               ("balanced clean Δ correct-decode", lambda r: pair(r, "clean", "correct_fraction")),
               ("flag (the test is outside it)", lambda r: _badge(r["flag_result"]))]
    key_columns = [("comparison", lambda k: f"{code(k[0])}<br>{code(k[1])} → {code(k[2])}"),
                   ("dtype", lambda k: code(k[3]))]
    return ("<h4 id=\"histogram_matching-harm\">Harm test on the unbalanced codebook (C7)</h4>"
            + mode_table(keys, by_mode, key_columns, columns, "tables/harm_test.csv", sources,
                         "Harm when either endpoint has Δ below −R_E on clean_unbalanced. It is reported beside the "
                         "flag with the balanced clean deltas, so the codebook's share is visible, and does not enter "
                         "the flag.", key_widths=(135, 72)))


def card_preconditions(method, sources):
    """Precondition status of the method's targeted conditions, per dtype and mode (item 3; C5 for sets)."""
    pre = sources.table("tables/preconditions.csv")
    if not len(pre):
        return "<p>This evaluation has no precondition checks.</p>"
    part = pre[pre.targeted_by.fillna("").str.split(",").apply(lambda names: method in names)]
    records = _records(part)
    key_of = lambda r: (r["check"], r["condition"], r["arm"], r["dtype"])  # noqa: E731
    keys = list(dict.fromkeys(key_of(r) for r in records))
    by_mode = {(r["threshold_mode"], key_of(r)): r for r in records}

    def status(r):
        verdict = "<span class=\"pass\">valid</span>" if _truthy(r["valid"]) else "<span class=\"fail\">fixture gap</span>"
        return (f"{verdict}<br><span class=\"range\">g max-F1 {fmt(r['g_max_f1'], 3)} vs R {fmt(r['range_max_f1'], 3)}; "
                f"g correct {fmt(r['g_correct_fraction'], 3)} vs R {fmt(r['range_correct_fraction'], 3)}</span>")

    key_columns = [("check", lambda k: code(k[0])), ("condition", lambda k: code(k[1])),
                   ("arm (reference → degraded)", lambda k: code(k[2])), ("dtype", lambda k: code(k[3]))]
    return mode_table(keys, by_mode, key_columns, [("precondition", status)], "tables/preconditions.csv", sources,
                      "A condition is a valid target when some endpoint degrades by g_E > R_E (strict), arm none "
                      "against clean; a failure is a fixture-gap row and leaves the condition out of this method's flag "
                      "for that dtype and mode. Its comparison rows are still reported.",
                      key_widths=(100, 130, 110, 72))


def card_caveats(method, manifest, sources, lookups):
    """Caveats: stated deviations, fixture gaps, not-assessable verdicts, grid-edge selections, MAD 0, anomalies."""
    card = CARDS[method]
    items = [esc(card["caveat"])] if card.get("caveat") else []
    gaps = sources.table("tables/fixture_gaps.csv")
    gaps = gaps if not len(gaps) else gaps[gaps.targeted_by.fillna("").str.split(",").apply(lambda names: method in names)]
    for mode in MODES:
        part = gaps[gaps.threshold_mode == mode] if len(gaps) else gaps
        if len(part):
            names = ", ".join(f"{r['condition']} {r['dtype']}" + (f" ({r['check']} {r['arm']})"
                                                                   if r["check"] != "fixture" else "")
                              for r in _records(part))
            items.append(f"<span class=\"fail\">Fixture gaps, {mode} mode</span>: {esc(names)} "
                         f"(<code>tables/fixture_gaps.csv</code>).")
    flags = lookups.flags[(lookups.flags.method == method) & (lookups.flags.level == "method")]
    for mode in MODES:
        counts = flags[flags.threshold_mode == mode].result.value_counts().to_dict()
        items.append(f"Verdicts, {mode} mode: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(
            counts.items())) + " (<code>tables/low_benefit_flags.csv</code>).")
    arms, conditions = set(), set()
    for m, c, before, after, targeted in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS:
        if m == method:
            arms |= {before, after}
            conditions |= set(targeted)
    conditions |= {"mf_density_clean"} if method == "sample_level_fitting" else {"clean", "combined"}
    conditions |= {c for c, _r in harness.calibrated_extra_roles(method, card["comparison"])}
    for mode in MODES:
        edges = sorted({(c, d, a, fmt(v[0])) for (s, c, d, a, m), v in lookups.selected.items()
                        if m == mode and v[1] and a in arms and c in conditions})
        if edges:
            items.append(f"Grid-edge selections, {mode} mode ({len(edges)}): " + esc("; ".join(
                f"{c} {d} {a} at {v}" for c, d, a, v in edges)) + " (<code>tables/operating_points.csv</code>).")
    zero = sorted({(r["condition"], r["dtype"], r["arm"]) for r in manifest.get("mad_zero_reference_round", [])
                   if r["arm"] in arms and r["condition"] in conditions})
    if zero:
        items.append(f"MAD 0 on a reference-round channel ({len(zero)}): " + esc("; ".join(
            f"{c} {d} {a}" for c, d, a in zero)) + "; the noise-mode cutoff is then the median whatever the value "
            "(<code>manifest.json</code>, <code>mad_zero_reference_round</code>).")
    anomalies = sorted({r["anomalies"] for r in _records(flags) if r.get("anomalies")})
    if anomalies:
        items.append("<span class=\"fail\">Anomalies</span>: " + esc("; ".join(anomalies)))
    return "<ul class=\"caveats\">" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def card_section(method, manifest, sources, lookups):
    card = CARDS[method]
    before, after, targeted = comparison_spec(method, card["comparison"])
    comparisons = [(c, b, a) for m, c, b, a, _t in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS
                   if m == method]
    downstream, direct = card_numbers(method, sources, lookups)
    all_targets = sorted({c for m, _c, _b, _a, t in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS
                          if m == method for c in t}, key=list(CALIBRATED_TEXT).index)
    harm = card_harm(sources) if method == "histogram_matching" else ""
    return (
        f"<section class=\"card\" id=\"card-{method}\"><h3>{esc(card['title'])}</h3>"
        f"<dl class=\"items\"><dt>What it does</dt><dd>{esc(card['does'])}</dd>"
        f"<dt>Problem</dt><dd>{esc(card['problem'])}</dd>"
        f"<dt>Comparisons</dt><dd>{'; '.join(f'{code(c)}: {code(b)} → {code(a)}' for c, b, a in comparisons)}</dd>"
        f"<dt>Targeted conditions</dt><dd>{', '.join(code(c) for c in all_targets)}; harm check "
        f"{code('mf_density_clean' if method == 'sample_level_fitting' else 'clean')}"
        f"{'' if method == 'sample_level_fitting' else '; combined reported'}.</dd>"
        f"<dt>Figures</dt><dd><a href=\"#fig-{method}\">{esc(card['title'])}: figures</a> (panels of "
        f"{code(card['comparison'])}: {code(before)} → {code(after)}).</dd></dl>"
        f"<h4 id=\"{method}-preconditions\">Precondition status of the targeted conditions</h4>"
        + card_preconditions(method, sources)
        + f"<h4 id=\"{method}-numbers\">Key numbers, both threshold modes side by side</h4>" + downstream + direct
        + f"<h4 id=\"{method}-flag\">Flag verdict per dtype and mode</h4>" + card_flags(method, lookups, sources)
        + harm + f"<h4 id=\"{method}-caveats\">Caveats</h4>" + card_caveats(method, manifest, sources, lookups)
        + "</section>")


def _recipe_steps(record):
    if record is None:
        return "no preprocessing"
    parts = []
    for step in record["steps"]:
        config = ", ".join(f"{k}={v}" for k, v in step["config"].items() if v is not None)
        parts.append(f"{code(step['step'])}({esc(config)})" + (f" → saved as {code(step['save_as'])}"
                                                               if step.get("save_as") else ""))
    extra = f"; extraction from {code(record['extraction_source'])}" if record.get("extraction_source") else ""
    return " → ".join(parts) + extra


def setup_section(manifest, sources):
    """Summary item 1: recipes, conditions with measured SNR, comparison types, metrics, preconditions and the run."""
    stats = sources.table("tables/image_statistics.csv")
    snr = {(r["condition"], r["dtype"], r["statistic"]): r for r in _records(
        stats[stats.statistic.isin(["snr_clutter_p50", "snr_pixel_p50"])])}
    recipes = "".join(f"<tr><td>{code(arm)}</td><td>{_recipe_steps(record)}</td></tr>"
                      for arm, record in manifest["recipes"].items())
    recipes += "".join(f"<tr><td>{code(arm)}</td><td>multi-FOV: recipe {code(spec['recipe'])} with "
                       f"<code>fit=\"{esc(spec['fit'])}\"</code></td></tr>"
                       for arm, spec in manifest.get("multi_fov_recipes", {}).items() if arm.endswith("_sample"))
    targets = harness.calibrated_targets()
    ranges = {s: harness.TARGET_RANGES[s] for s in ("snr_clutter_p50", "snr_pixel_p50")}

    def snr_cell(condition, statistic):
        cells = []
        for dtype in harness.DTYPES:
            r = snr.get((condition, dtype, statistic))
            if r is None:
                continue
            flag = "" if _truthy(r["within_target_range"]) else " <span class=\"fail\">(outside)</span>"
            cells.append(f"{dtype} {fmt(r['value'], 3)}{flag}")
        return "<br>".join(cells) or "–"

    conditions = "".join(
        f"<tr><td>{code(c)}</td><td>{esc(text)}</td><td>{esc(', '.join(targets.get(c, [])) or '–')}</td>"
        f"<td class=\"num\">{snr_cell(c, 'snr_clutter_p50')}</td><td class=\"num\">{snr_cell(c, 'snr_pixel_p50')}"
        f"</td></tr>" for c, text in CALIBRATED_TEXT.items())
    k = manifest.get("saturation_k") or {}
    k_text = "; ".join(f"{d}: k = {v['k']:.4g} (j = {v['j']}), mean clipped fraction {v['mean_clipped_fraction']:.3g} "
                       "on development seeds" for d, v in k.items())
    pre = sources.table("tables/preconditions.csv")
    pre = pre if len(pre) else pd.DataFrame(columns=["check", "condition", "arm", "dtype", "threshold_mode"])
    columns = [(d, m) for d in harness.DTYPES for m in MODES if ((pre.dtype == d) & (pre.threshold_mode == m)).any()]
    cells = {(r["check"], r["condition"], r["arm"], r["dtype"], r["threshold_mode"]): r for r in _records(pre)}
    rows = list(dict.fromkeys((r["check"], r["condition"], r["arm"]) for r in _records(pre)))

    def pre_cell(r):
        if r is None:
            return "–"
        if _truthy(r["valid"]):
            return "<span class=\"pass\">pass</span>"
        return (f"<span class=\"fail\">FAIL</span> <span class=\"range\">g {fmt(r['g_max_f1'], 2)}/"
                f"{fmt(r['g_correct_fraction'], 2)} ≤ R {fmt(r['range_max_f1'], 2)}/"
                f"{fmt(r['range_correct_fraction'], 2)}</span>")

    matrix = ("<div class=\"scroll\"><table class=\"fit\">" + _colgroup([100, 130, 90] + [None] * len(columns))
              + "<thead><tr><th>check</th><th>condition</th><th>arm</th>"
              + "".join(f"<th class=\"mode-{m}\">{d}, {m} mode</th>" for d, m in columns) + "</tr></thead><tbody>"
              + "".join(f"<tr><td>{code(c)}</td><td>{code(n)}</td><td>{code(a)}</td>"
                        + "".join(f"<td>{pre_cell(cells.get((c, n, a, d, m)))}</td>" for d, m in columns) + "</tr>"
                        for c, n, a in rows) + "</tbody></table></div>")
    parameters = manifest["parameters"]
    modes = "; ".join(f"{m}: grid {{{', '.join(fmt(v) for v in s['grid'])}}}, fixed {', '.join(fmt(v) for v in s['fixed'])}"
                      for m, s in parameters["threshold_modes"].items())
    time_v, projection = manifest.get("time_v", {}), manifest.get("projection") or {}
    plans = "; ".join(f"{label}: {fmt(plan.get('projected_seconds'))} s" for label, plan in
                      (projection.get("plans") or {}).items())
    plan = manifest["plan"]["run"]
    return f"""
<section id="setup"><h2>1. Setup</h2>
<p class="note"><strong>Qualification.</strong> {esc(manifest.get('qualification', ''))}. Every scene is a calibrated
synthetic development preset (<code>calibrated-development-v1</code>, W-241) whose appearance was matched to the
W-238 development targets measured on processed uint8 exports; <strong>it is not D04</strong>, no real data were used,
<strong>no preprocessing default is recommended</strong>, and the low-benefit flag and its 0.02 threshold are
<strong>provisional</strong>. The uint16 scale (× 16) is unverified against real data.</p>
<h3 id="setup-recipes">Recipes and steps (page defaults)</h3>
<div class="scroll"><table class="fit">{_colgroup([130, None])}<thead><tr><th>arm</th><th>steps</th></tr></thead><tbody>{recipes}</tbody></table></div>
<p class="caption">Source: {code('manifest.json')} (<code>recipes</code>, <code>multi_fov_recipes</code>). r_z is capped
at 3 (C4). The pipeline default remains recipe 1; this report recommends none.</p>
<h3 id="setup-conditions">Conditions and their measured SNR</h3>
<div class="scroll"><table class="fit">{_colgroup([140, None, 190, 120, 120])}<thead><tr><th>condition</th>
<th>definition</th><th>targeted by</th>
<th>clutter SNR p50</th><th>pixel SNR p50</th></tr></thead><tbody>{conditions}</tbody></table></div>
<p class="caption">Source: {code('tables/image_statistics.csv')} (W-238 tool on the development seeds, adaptive
selection, pooled per condition or set). W-238 development ranges: clutter SNR p50
{fmt(ranges['snr_clutter_p50'][0], 3)}–{fmt(ranges['snr_clutter_p50'][1], 3)}, pixel SNR p50
{fmt(ranges['snr_pixel_p50'][0], 3)}–{fmt(ranges['snr_pixel_p50'][1], 3)}. Saturation: {esc(k_text) or '–'}.
Scenes are {code('×'.join(str(v) for v in manifest['scenes'][0]['shape_zyx']))} voxels (ZYX), four rounds and four
channels, 80 amplicons, the balanced 16-gene codebook unless stated.</p>
<h3 id="setup-comparisons">Comparison types</h3>
<dl class="items">{''.join(f'<dt>{esc(n)}</dt><dd>{esc(t)}</dd>' for n, t in COMPARISON_TYPES)}</dl>
<p>Each comparison runs on its targeted conditions, on the clean harm check and on combined (reported, outside the
flag), in each dtype that was run, in both threshold modes.</p>
<h3 id="setup-metrics">What each metric measures</h3>
<dl class="items">{''.join(f'<dt>{esc(n)}</dt><dd>{esc(t)}</dd>' for n, t in METRIC_TEXT)}</dl>
<h3 id="setup-preconditions">Precondition results per condition, dtype and mode</h3>
<p>Item 3: arm <code>none</code> on the condition against <code>none</code> on clean (multi-FOV sets: against
<code>mf_density_clean</code>, pooled), and for the sets the set-specific check of each recipe's per-FOV-fitted arm
(C5). g is the mean degradation of max-F1/correct-decode over the held-out seeds and R the larger seed range; a
condition passes when some g exceeds its R strictly.</p>
{matrix}
<p class="caption">Source: {code('tables/preconditions.csv')} (sha256
{code(sources.records['tables/preconditions.csv']['sha256'][:16])}…); failures are the rows of
{code('tables/fixture_gaps.csv')}.</p>
<h3 id="setup-run">Seeds, dtypes, modes, grids and the run</h3>
<table><tbody>
<tr><th>Seeds</th><td>development {', '.join(map(str, parameters['development_seeds']))} (operating-point
selection, statistics, saturation k); held-out {', '.join(map(str, parameters['evaluation_seeds']))} (every
reported endpoint)</td></tr>
<tr><th>Dtypes</th><td>uint8 for every condition; uint16 only for {esc(', '.join(plan['uint16_conditions']))} (C2);
multi-FOV sets in {esc(', '.join(plan['multi_fov_dtypes']))}</td></tr>
<tr><th>Threshold modes</th><td>{esc(modes)}; operating point per mode, condition, dtype and arm: the highest mean F1
over the development seeds, ties to the smaller value, grid ends marked</td></tr>
<tr><th>Reductions</th><td>{esc(', '.join(manifest.get('reductions') or []) or 'none applied')}; selected plan
{esc(projection.get('selected_plan', '–'))}; pilot projections {esc(plans or '–')} (budget
{fmt(projection.get('budget_seconds'))} s)</td></tr>
<tr><th>Wall time</th><td>{fmt(time_v.get('wall_seconds'))} s, maximum RSS {fmt(time_v.get('max_rss_kib'))} KiB, exit
status {fmt(time_v.get('exit_status'))}; {len(manifest['scenes'])} single-FOV scenes,
{len(manifest.get('multi_fov_scenes', []))} multi-FOV FOV scenes, {len(manifest['runs'])} processed runs</td></tr>
</tbody></table>
<p class="caption">Source: {code('manifest.json')} (<code>parameters</code>, <code>plan</code>,
<code>reductions</code>, <code>projection</code>, <code>time_v</code>).</p>
</section>"""


def findings(manifest, sources):
    """Summary item 3: (label, text, sources) for each cross-cutting finding and hypothesis, from the tables."""
    out = []
    pre = sources.table("tables/preconditions.csv")
    if not len(pre):
        pre = pd.DataFrame(columns=["check", "condition", "arm", "dtype", "threshold_mode", "valid", "reference",
                                    "range_max_f1"])
    failed = pre[~pre.valid.map(_truthy).astype(bool)]
    per_mode = ", ".join(f"{m} {int((failed.threshold_mode == m).sum())} of {int((pre.threshold_mode == m).sum())}"
                         for m in MODES)
    out.append(("finding", f"{len(failed)} of {len(pre)} precondition checks failed ({per_mode}); every failure is a "
                           "fixture-gap row and leaves the condition out of the flags of the methods that target it.",
                ["tables/preconditions.csv", "tables/fixture_gaps.csv"]))
    valid = {(r["check"], r["condition"], r["arm"], r["dtype"], r["threshold_mode"]): _truthy(r["valid"])
             for r in _records(pre)}
    differ = sorted({(c, n, a, d) for (c, n, a, d, m), v in valid.items()
                     if all((c, n, a, d, o) in valid for o in MODES) and valid[(c, n, a, d, MODES[0])]
                     != valid[(c, n, a, d, MODES[1])]})
    if differ:
        out.append(("finding", f"The precondition verdict differs between the modes for {len(differ)} checks: "
                    + "; ".join(f"{n} {d}" + (f" ({c} {a})" if c != "fixture" else "") + " passes in "
                                + next(m for m in MODES if valid[(c, n, a, d, m)]) + " mode only"
                                for c, n, a, d in differ) + ".", ["tables/preconditions.csv"]))
    flags = sources.table("tables/low_benefit_flags.csv")
    method_rows = flags[flags.level == "method"]
    counts = "; ".join(f"{m} mode: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(
        method_rows[method_rows.threshold_mode == m].result.value_counts().to_dict().items())) for m in MODES)
    none = "" if (method_rows.result == "not_flagged").any() else " No comparison is not flagged in either mode."
    out.append(("finding", f"Flag verdicts over {len(method_rows)} comparison × dtype × mode rows: {counts}.{none}",
                ["tables/low_benefit_flags.csv"]))
    clean = flags[(flags.level == "clean") & ~flags.clean_holds.map(_truthy)]
    for method, group in clean.groupby("method", sort=False):
        deltas = group.delta_max_f1.astype(float)
        out.append(("finding", f"{method}: clean does not hold in {len(group)} rows ("
                    + ", ".join(sorted({f'{c} {d} {m}' for c, d, m in zip(group.comparison, group.dtype,
                                                                          group.threshold_mode)}))
                    + f"); Δ max-F1 on clean {fmt(deltas.min(), 3)} to {fmt(deltas.max(), 3)}.",
                    ["tables/low_benefit_flags.csv"]))
    harm = sources.table("tables/harm_test.csv")
    if len(harm):
        harmed = harm[harm.harm.map(_truthy)]
        out.append(("finding", f"Histogram matching's unbalanced-codebook harm test shows harm in {len(harmed)} of "
                    f"{len(harm)} comparison × dtype × mode rows ("
                    + "; ".join(f"{r['comparison']} {r['dtype']} {r['threshold_mode']}: {r['harm_endpoints']}"
                                for r in _records(harmed)) + "); it stays outside the flag (C7).",
                    ["tables/harm_test.csv"]))
    points = sources.table("tables/operating_points.csv")
    dev = points[points.operating_point == "dev_max_f1"]
    edges = dev[dev.at_grid_edge.map(_truthy)]
    detail = "; ".join(f"{m} at {fmt(v)}: {n}" for (m, v), n in edges.groupby(["threshold_mode", "threshold"]).size()
                       .items())
    out.append(("finding", f"Anomaly: {len(edges)} of {len(dev)} development-selected operating points sit at a grid "
                           f"edge ({detail}); the grids were not extended.", ["tables/operating_points.csv"]))
    runs = pd.DataFrame([r for r in manifest["runs"] if r.get("condition")])
    if len(runs):
        none = runs[runs.arm == "none"].set_index(["condition", "dtype", "seed"]).output_sha256
        rest = runs[runs.arm != "none"]
        rest = rest.assign(same=rest.output_sha256.to_numpy() == none.reindex(pd.MultiIndex.from_frame(
            rest[["condition", "dtype", "seed"]])).to_numpy())
        same = rest.groupby("arm", sort=False).same.agg(["sum", "size"])
        same = same[same["sum"] > 0]
        if len(same):
            where = {arm: sorted({f"{c} {d}" for c, d in zip(g.condition, g.dtype)})
                     for arm, g in rest[rest.same].groupby("arm")}
            out.append(("finding", "Anomaly: some arms return the input unchanged, so their isolated Δ against none is "
                        "0 by construction; outputs equal to none byte for byte: "
                        + "; ".join(f"{arm} on {int(r['sum'])} of {int(r['size'])} single-FOV scenes"
                                    + (f" ({', '.join(where[arm])})" if len(where[arm]) <= 3 else "")
                                    for arm, r in same.iterrows()) + ".", ["manifest.json"]))
    diagnostics = _with_sets(sources.table("tables/diagnostics.csv"))
    reference = diagnostics[(diagnostics["round"] == diagnostics["round"].iloc[0]) & (diagnostics.arm == "none")
                            & diagnostics.multi_fov.isna()]
    if len(reference):
        many = int((reference.zero_fraction.astype(float) >= 0.1).sum())
        out.append(("finding", f"The input (arm none) has a zero fraction of at least 0.10 on {many} of "
                               f"{len(reference)} single-FOV reference-round channels (minimum "
                               f"{fmt(reference.zero_fraction.min(), 3)}), because the generator clips negative noise "
                               "to 0; where it does, the 10th percentile, and so the scalar background estimate, is 0.",
                    ["tables/diagnostics.csv"]))
    zero = manifest.get("mad_zero_reference_round", [])
    arms = sorted({r["arm"] for r in zero})
    out.append(("finding", f"Anomaly: {len(zero)} condition × dtype × arm entries have a reference-round channel with "
                           f"MAD 0 (arms {', '.join(arms)}); there the noise-mode cutoff equals the median and the "
                           "noise grid does not change detections.", ["manifest.json", "tables/diagnostics.csv"]))
    anomalies = method_rows[method_rows.anomalies.notna() & (method_rows.anomalies.astype(str) != "")]
    stats = sources.table("tables/image_statistics.csv")
    undefined = stats[stats.value.isna()]
    out.append(("finding", f"Anomaly: {len(anomalies)} flag rows record an undefined endpoint; "
                           f"{len(undefined)} image statistics are undefined ("
                           + ", ".join(sorted(set(undefined.statistic))) + ").",
                ["tables/low_benefit_flags.csv", "tables/image_statistics.csv"]))
    scenes = pd.DataFrame(manifest["scenes"])
    clipped = scenes.groupby(["condition", "dtype"]).clipped_fraction.mean()
    designed = {("saturation", "uint8"), ("bright_outliers", "uint8")}
    designed = list(designed)
    unexpected = clipped[(clipped > 1e-4) & ~clipped.index.isin(designed)]
    out.append(("finding", "Clipping above the dtype maximum at generation, mean over seeds: "
                + "; ".join(f"{c} {d} {v:.2g}" for (c, d), v in clipped[clipped.index.isin(designed)].items())
                + " by design; " + ("unexpected: " + "; ".join(f"{c} {d} {v:.2g}" for (c, d), v in unexpected.items())
                                    if len(unexpected) else "no other condition above 10⁻⁴") + ".",
                ["manifest.json"]))
    defined = stats[stats.value.notna()]
    outside = defined[~defined.within_target_range.map(_truthy)]
    top = outside.statistic.value_counts().head(4)
    out.append(("finding", f"{len(outside)} of {len(defined)} defined image statistics lie outside the W-238 "
                           "development ranges; most often " + ", ".join(f"{s} ({n})" for s, n in top.items()) + ".",
                ["tables/image_statistics.csv"]))
    fixture = pre[(pre.check == "fixture") & (pre.reference == "clean")]
    if len(failed):
        spans = fixture.range_max_f1.astype(float)
        out.append(("hypothesis", f"With three held-out seeds of 80 amplicons, the max-F1 seed range R of the fixture "
                                  f"checks is {fmt(spans.min(), 2)}–{fmt(spans.max(), 2)}, larger than most single-factor "
                                  "degradations g; the calibrated factors may be mild relative to seed-to-seed "
                                  "variation, so more seeds or stronger factors could turn fixture gaps into valid "
                                  "targets. This run does not test it.", ["tables/preconditions.csv"]))
    zero_fraction = defined[defined.statistic == "zero_fraction_p50"]
    if len(zero_fraction) and not zero_fraction.within_target_range.map(_truthy).any():
        low, high = harness.TARGET_RANGES["zero_fraction_p50"]
        out.append(("hypothesis", f"Every condition's zero fraction p50 ({fmt(zero_fraction.value.min(), 2)}–"
                                  f"{fmt(zero_fraction.value.max(), 2)}) is below the W-238 range ({fmt(low, 2)}–"
                                  f"{fmt(high, 2)}), and real volumes have MAD 0 in 77 % of cases (T2); the noise-mode "
                                  "results here may therefore not transfer to real exports. Untested.",
                    ["tables/image_statistics.csv"]))
    if (clean.method == "percentile_normalization").any():
        out.append(("hypothesis", "Percentile normalization loses max-F1 on clean in both modes; its p_low offset and "
                                  "the saturation of the brightest 0.1 % after stretching are candidate causes (see the "
                                  "clipped and saturated fractions on its card). This run does not isolate the cause.",
                    ["tables/comparisons.csv", "tables/low_benefit_flags.csv"]))
    return out


def findings_section(manifest, sources):
    items = []
    for label, text, names in findings(manifest, sources):
        for name in names:
            sources.used.add(name)
        items.append(f"<li><span class=\"label {label}\">{label}</span> {esc(text)} <span class=\"range\">Source: "
                     + ", ".join(code(n) for n in names) + "</span></li>")
    return ("<section id=\"findings\"><h2>3. Cross-cutting findings and anomalies</h2><p>Each item is a "
            "<span class=\"label finding\">finding</span> shown by the named table or a "
            "<span class=\"label hypothesis\">hypothesis</span>, an explanation this run does not test. Counts are "
            "read from the tables; nothing is pooled across threshold modes.</p><ul class=\"caveats\">"
            + "".join(items) + "</ul></section>")


FIGURE_ANCHORS = (("panels", "Before, after and difference, cropped around the dim punctum (changes 1, 2 and 9)"),
                  ("overlays", "Detections against truth (change 3)"),
                  ("colour", "Colour-calling view (change 4)"),
                  ("background", "Background estimate against truth (change 5)"),
                  ("histograms", "Histograms of the signal channels (change 6)"),
                  ("dots", "Δ dot plot with the provisional flag band (change 7)"),
                  ("pr", "Precision–recall small multiples (change 8)"))

APPENDIX = (("identity", "Identity, revision and commands"), ("render-checks", "Render checks"),
            ("sources", "Sources and checksums"), ("a-projection", "Reductions, projection and timing"),
            ("a-saturation", "Saturation k search"), ("a-preconditions", "Preconditions (full)"),
            ("a-fixture-gaps", "Fixture gaps"), ("a-flags", "Low-benefit flags (full)"),
            ("a-harm", "Harm test"), ("a-comparisons", "Comparisons per dtype"),
            ("a-operating-points", "Operating points per dtype"), ("a-endpoints", "Held-out endpoints per seed"),
            ("a-per-seed", "Max-F1 and AUPRC per seed"), ("a-direct", "Direct metrics per seed"),
            ("a-multi-fov", "Multi-FOV tables"), ("a-diagnostics", "Diagnostics, reference round"),
            ("a-statistics", "Image statistics"), ("a-mad-zero", "MAD-zero entries"),
            ("a-verification", "Threshold-subset verification"))


def reading_order_section():
    cards = "".join(f"<li><a href=\"#card-{m}\">{esc(c['title'])}</a> → <a href=\"#fig-{m}\">figures</a></li>"
                    for m, c in CARDS.items())
    appendix = "".join(f"<li><a href=\"#{a}\">{esc(t)}</a></li>" for a, t in APPENDIX)
    return f"""<section id="reading-order"><h2>4. Reading order</h2><ol>
<li><a href="#setup">Setup</a>: recipes, <a href="#setup-conditions">conditions and SNR</a>,
<a href="#setup-preconditions">precondition results</a> and the <a href="#setup-run">run</a>.</li>
<li><a href="#cards">Method cards</a>, in page order, each followed by its figures:<ol>{cards}</ol></li>
<li><a href="#findings">Cross-cutting findings and anomalies</a>.</li>
<li><a href="#method-figures">Method sections with figures</a>; in each, the panels come first, then detections,
colour calling, background, histograms, the Δ dot plot and the PR small multiples.</li>
<li><a href="#appendix">Appendix</a>, for audit:<ol>{appendix}</ol></li></ol></section>"""


def figure_block(method, anchor, title, png, caption):
    return (f"<h3 id=\"fig-{method}-{anchor}\">{esc(title)}</h3><figure>{image(png, title)}"
            f"<figcaption>{caption}</figcaption></figure>")


def figures_section(method, figure, sources):
    card = CARDS[method]
    before, after, _targeted = comparison_spec(method, card["comparison"])
    rows = figure["rows"]
    where = ", ".join(_row_title(r) for r in rows)
    common = (f"{code(DISPLAY_DTYPE)}, held-out seed {DISPLAY_SEED}, reference round; rows: {esc(where)}.")
    parts = [figure_block(method, "panels", FIGURE_ANCHORS[0][1], figure["panels"],
                          f"{code(card['comparison'])}: {code(before)} (before) → {code(figure['labels']['after'])} "
                          f"(after); {common} Each row shows the Z slice and the XZ view through its dim punctum (red "
                          f"square; lowest realized peak), cropped to ±{CROP_YX} voxels in X and Y with every Z plane, in "
                          "one linear display range shared by before and after (stated on each panel), and after − "
                          "before in a symmetric diverging range. Cyan circles are truth puncta on that channel within "
                          "one voxel of the plane. The image panels do not depend on the threshold mode and are drawn "
                          "once. The line profile is drawn in two adjacent panels, noise mode then adaptive mode, each "
                          "marking that mode's detection cutoffs (dashed) at its development-selected value; an "
                          "extraction snapshot has no detection cutoff.")]
    parts.append(figure_block(method, "overlays", FIGURE_ANCHORS[1][1], figure["overlays"],
                              f"Maximum projection over Z and channels of each arm's detection image; {common} Noise "
                              "mode (left pair) and adaptive mode (right pair) side by side, each at its own "
                              "development-selected threshold from <code>tables/operating_points.csv</code>. Green "
                              "circles: true positives; red crosses: false positives; yellow squares: missed truth "
                              "puncta (greedy matching within 2 voxels). Each panel's detection was rerun on the verified "
                              "image and its spot, match and read counts and cutoffs equal the saved curve row (see "
                              "<a href=\"#render-checks\">Render checks</a>)."))
    parts.append(figure_block(method, "colour", FIGURE_ANCHORS[2][1], figure["colour"],
                              f"Targeted row {esc(_row_title(rows[0]))}: channel × round neighbourhood sums (the "
                              "extraction box, radius (1, 2, 2)) at the reference-round centre of the dim punctum and of "
                              "the puncta at the 25th, 50th and 90th percentiles of the realized peak "
                              f"({esc(', '.join(figure['puncta']))}); red boxes mark the true colour of each round. Before "
                              f"reads {code(before)}; after reads {code(figure['labels']['after'])}. The intensity "
                              "vectors do not depend on the threshold mode and are drawn once (first two columns); the "
                              "calls depend on it and are shown in two adjacent panels per punctum, noise mode then "
                              "adaptive mode (observed sequence and status, or not detected, before and after)."))
    if card.get("background"):
        parts.append(figure_block(method, "background", FIGURE_ANCHORS[3][1], figure["background"],
                                  f"Estimate = input − the {code(after)} bg_corrected snapshot, against the background "
                                  f"truth (including the pedestal), on the dim punctum's Z slice and channel; {common} "
                                  "Truth and estimate share one display range; the residual is estimate − truth. The "
                                  "background estimate is computed before detection, so it is identical in both "
                                  "threshold modes and is drawn once."))
    parts.append(figure_block(method, "histograms", FIGURE_ANCHORS[4][1], figure["histograms"],
                              f"Every voxel of each signal-carrying channel (at least one truth punctum on it in the "
                              f"reference round; {esc(figure['channels_note'])}), log scale; {common} Each channel has "
                              "two adjacent panels, noise mode then adaptive mode. Dashed: the noise cutoff at value 5 "
                              "from <code>tables/diagnostics.csv</code> (checked against the image). Solid: that mode's "
                              "detection cutoff at its development-selected value (the cutoffs of the saved curve row). "
                              + ("The extraction snapshot is not detected on, so its row is drawn once with the noise "
                                 "cutoff only, and a companion row shows the xsrc arm's detection image with both modes' "
                                 "detection cutoffs, so every line sits on the intensity scale it applies to."
                                 if card.get("snapshot") else "")))
    parts.append(figure_block(method, "dots", FIGURE_ANCHORS[5][1], figure["dots"],
                              "Every comparison of the method on its targeted conditions, clean, combined and the "
                              "reported rows, in both dtypes; noise mode (left pair) and adaptive mode (right pair). Dots: "
                              "mean paired Δ over the held-out seeds; bars: min to max of the per-seed Δ; shaded: the "
                              "provisional flag band ±max(R_E, 0.02) of the row. Hollow dots are targeted conditions "
                              "outside T* (fixture gaps). Source: <code>tables/comparisons.csv</code>."))
    source = "curves/multi_fov_curves.csv (pooled over each set's FOVs per seed)" if method == "sample_level_fitting" \
        else "curves/pr_curves.csv"
    collapsed_note = ("Collapsed curves: " + esc("; ".join(figure["collapsed"])) + ".") if figure["collapsed"] else \
        "No curve of this pair collapses to one point."
    parts.append(figure_block(method, "pr", FIGURE_ANCHORS[6][1], figure["pr"],
                              f"Only the {code(card['comparison'])} pair {code(before)} (grey) and {code(after)} "
                              "(orange); one panel per condition and dtype, noise mode left and adaptive mode right. "
                              f"Mean precision and recall over the held-out seeds at each grid value ({esc(source)}); "
                              "the end labels are the lowest and highest values. A curve whose grid values all give "
                              "the same point is kept and drawn as a star, labelled MAD 0 where the manifest records a "
                              f"MAD-0 channel for that arm in noise mode. {collapsed_note}"))
    return (f"<section id=\"fig-{method}\"><h2>Figures: {esc(card['title'])}</h2>"
            f"<p><a href=\"#card-{method}\">Back to the card</a>.</p>" + "".join(parts) + "</section>")


#: Appendix tables are laid out to fit one 1400 × 4000 px capture per piece: column groups within
#: FULL_WIDTH_PX (key columns repeated) and row blocks within FULL_BLOCK_PX, estimated conservatively at the
#: 9.5 px table font (CHAR_PX per character plus CELL_PX padding per cell, LINE_PX per text line).
FULL_WIDTH_PX = 1300
FULL_BLOCK_PX = 3600
CHAR_PX = 6.3
CELL_PX = 10
WRAP_CHARS = 36
LINE_PX = 16
KEY_COLUMNS = ("method", "comparison", "check", "condition", "multi_fov", "role", "level", "dtype", "arm", "seed",
               "fov_id", "threshold_mode", "operating_point", "round", "channel", "statistic", "plan", "item", "mode",
               "j")


def _lines(lengths, px):
    """Wrapped line count of texts of the given lengths in a column of px pixels."""
    return np.maximum(1, np.ceil(np.asarray(lengths, dtype=float) * CHAR_PX / max(px - CELL_PX, 1)))


def full_table(frame, source, sources, caption="", anchor=None, blocks=True):
    """Every column and row of a saved table (numbers rounded for display only), naming its checked source.

    Wide tables are split into column groups that fit the page (the key columns repeat in each), and with
    blocks, long tables into row blocks that each fit one capture; each piece gets the sub-anchor
    <anchor>-r<i>c<j> when there is more than one.
    """
    frame = frame.reset_index(drop=True)
    columns = list(frame.columns)
    texts = {c: [fmt(v) for v in frame[c].astype(object).where(frame[c].notna(), None)] for c in columns}
    lengths = {c: np.asarray([len(t) for t in texts[c]] or [0]) for c in columns}
    widths = {c: int(min(max(int(lengths[c].max()), min(len(str(c)), 7)), WRAP_CHARS) * CHAR_PX + CELL_PX)
              for c in columns}
    keys = [c for c in columns if c in KEY_COLUMNS]
    chunks, current, used = [], [], sum(widths[c] for c in keys)
    for c in columns:
        if c in keys:
            continue
        if current and used + widths[c] > FULL_WIDTH_PX:
            chunks.append(current)
            current, used = [], sum(widths[k] for k in keys)
        current.append(c)
        used += widths[c]
    chunks.append(current)
    chunks = [keys + chunk for chunk in chunks]
    header_px = max(LINE_PX * int(_lines([len(str(c))], widths[c]).max()) for c in columns) if columns else LINE_PX
    heights = (LINE_PX * np.max([_lines(lengths[c], widths[c]) for c in columns], axis=0)
               if columns and len(frame) else np.zeros(len(frame)))
    bounds, start, height = [], 0, 0.0
    for i, h in enumerate(heights):
        if blocks and i > start and height + h > FULL_BLOCK_PX - header_px:
            bounds.append((start, i))
            start, height = i, 0.0
        height += h
    bounds.append((start, len(frame)))
    digest = sources.records[source]["sha256"]
    parts = []
    for i, (first, last) in enumerate(bounds):
        for j, chunk in enumerate(chunks):
            if len(bounds) > 1 or len(chunks) > 1:
                where = f"rows {first + 1}–{last} of {len(frame)}" + (
                    f", column group {j + 1} of {len(chunks)} ({esc(chunk[len(keys)])} … {esc(chunk[-1])})"
                    if len(chunks) > 1 and len(chunk) > len(keys) else "")
                ident = f" id=\"{anchor}-r{i + 1}c{j + 1}\"" if anchor else ""
                parts.append(f"<h4{ident}>{where}</h4>")
            head = "".join(f"<th>{esc(c)}</th>" for c in chunk)
            body = "".join("<tr>" + "".join(f"<td>{esc(texts[c][r])}</td>" for c in chunk) + "</tr>"
                           for r in range(first, last))
            width = sum(widths[c] for c in chunk)
            parts.append(f"<div class=\"scroll\"><table class=\"full\" style=\"width:{width}px\">"
                         f"{_colgroup([widths[c] for c in chunk])}<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
                         "</table></div>")
    return "".join(parts) + (f"<p class=\"caption\">Source: {code(source)} (sha256 {code(digest[:16])}…), "
                             f"{len(frame)} rows. {caption}</p>")


def appendix_tables(manifest, sources):
    parts = []

    def per_dtype(anchor, title, source, caption=""):
        frame = sources.table(source)
        block = f"<section id=\"{anchor}\"><h2>A. {esc(title)}</h2>"
        for dtype in harness.DTYPES:
            part = frame[frame.dtype == dtype]
            if len(part):
                block += (f"<h3 id=\"{anchor}-{dtype}\">{dtype}</h3>"
                          + full_table(part, source, sources, caption, anchor=f"{anchor}-{dtype}"))
        parts.append(block + "</section>")

    def whole(anchor, title, source, caption=""):
        parts.append(f"<section id=\"{anchor}\"><h2>A. {esc(title)}</h2>"
                     + full_table(sources.table(source), source, sources, caption, anchor=anchor) + "</section>")

    whole("a-preconditions", "Preconditions (full)", "tables/preconditions.csv")
    whole("a-fixture-gaps", "Fixture gaps", "tables/fixture_gaps.csv")
    whole("a-flags", "Low-benefit flags (full)", "tables/low_benefit_flags.csv",
          "Provisional rule and 0.02 threshold; one row per targeted condition, one clean row and one method row.")
    whole("a-harm", "Harm test", "tables/harm_test.csv")
    per_dtype("a-comparisons", "Comparisons per dtype", "tables/comparisons.csv",
              "Held-out mean, min and max of before, after and Δ; one row per mode.")
    per_dtype("a-operating-points", "Operating points per dtype", "tables/operating_points.csv")
    per_dtype("a-endpoints", "Held-out endpoints per seed", "tables/endpoints.csv")
    per_dtype("a-per-seed", "Max-F1 and AUPRC per seed", "tables/recipe_per_seed.csv")
    per_dtype("a-direct", "Direct metrics per seed", "tables/direct_metrics.csv")
    block = "<section id=\"a-multi-fov\"><h2>A. Multi-FOV tables</h2>"
    for source in ("tables/multi_fov_endpoints.csv", "tables/multi_fov_per_fov_endpoints.csv",
                   "tables/multi_fov_spread.csv"):
        if source in sources.records:
            anchor = "a-multi-fov-" + Path(source).stem.replace("multi_fov_", "").replace("_", "-")
            block += (f"<h3 id=\"{anchor}\">{code(source)}</h3>"
                      + full_table(sources.table(source), source, sources, anchor=anchor))
    parts.append(block + "</section>")
    diagnostics = sources.table("tables/diagnostics.csv")
    block = ("<section id=\"a-diagnostics\"><h2>A. Diagnostics, every round</h2><p>Every saved row of "
             f"{code('tables/diagnostics.csv')} ({len(diagnostics)} rows), one table per dtype and round. Each table "
             "is collapsed; open it to read its rows.</p>")
    for dtype in harness.DTYPES:
        part = diagnostics[diagnostics.dtype == dtype]
        if not len(part):
            continue
        block += f"<h3 id=\"a-diagnostics-{dtype}\">{dtype}</h3>"
        for round_label, rows in part.groupby("round", sort=True):
            block += (f"<details><summary>{dtype}, {esc(round_label)}: {len(rows)} rows</summary>"
                      + full_table(rows, "tables/diagnostics.csv", sources, blocks=False) + "</details>")
    parts.append(block + "</section>")
    whole("a-statistics", "Image statistics", "tables/image_statistics.csv")
    zero = pd.DataFrame(manifest.get("mad_zero_reference_round", []))
    parts.append("<section id=\"a-mad-zero\"><h2>A. MAD-zero entries</h2>"
                 + (full_table(zero, "manifest.json", sources, "From the manifest's mad_zero_reference_round list.",
                               anchor="a-mad-zero") if len(zero) else "<p>None.</p>") + "</section>")
    verification = pd.DataFrame([dict(mode=m, **{k: (", ".join(map(str, v)) if isinstance(v, list) else v)
                                                 for k, v in r.items()})
                                 for m, r in manifest.get("threshold_subset_verification", {}).items()])
    parts.append("<section id=\"a-verification\"><h2>A. Threshold-subset verification</h2>"
                 + full_table(verification, "manifest.json", sources,
                              "From the manifest's threshold_subset_verification record.", anchor="a-verification")
                 + "</section>")
    return "".join(parts)


def projection_section(manifest, sources):
    projection = manifest.get("projection") or {}
    plans = pd.DataFrame([dict(plan=label, **{k: v for k, v in plan.items() if not isinstance(v, (dict, list))})
                          for label, plan in (projection.get("plans") or {}).items()])
    pilot = projection.get("pilot") or {}
    timing = pd.DataFrame([dict(item=k, seconds=v if not isinstance(v, dict) else json.dumps(v))
                           for k, v in manifest.get("timing_seconds", {}).items()])
    raw = "\n".join(manifest.get("time_v", {}).get("raw", []))
    return (f"<section id=\"a-projection\"><h2>A. Reductions, projection and timing</h2>"
            f"<p>Reductions applied: {esc(', '.join(manifest.get('reductions') or []) or 'none')}; selected plan "
            f"{esc(projection.get('selected_plan', '–'))}; budget {fmt(projection.get('budget_seconds'))} s. "
            f"{esc(projection.get('basis', ''))}</p>"
            + (full_table(plans, "manifest.json", sources, "Pilot projections of the full matrix before and after R3.",
                          anchor="a-projection-plans") if len(plans) else "")
            + (f"<p>Pilot record: {esc(json.dumps({k: v for k, v in pilot.items() if not isinstance(v, (dict, list))}))}"
               "</p>" if pilot else "")
            + (full_table(timing, "manifest.json", sources, "The run's timing_seconds record.",
                          anchor="a-projection-timing") if len(timing) else "")
            + f"<h3 id=\"a-projection-time\">/usr/bin/time -v of the full run</h3><pre>{esc(raw)}</pre></section>")


def saturation_section(manifest, sources):
    rows = []
    for dtype, record in (manifest.get("saturation_k") or {}).items():
        for step in record.get("trace", []):
            rows.append(dict(dtype=dtype, **step, selected=step["j"] == record["j"]))
    held = [dict(condition=s["condition"], dtype=s["dtype"], seed=s["seed"], split=s["split"],
                 clipped_fraction=s["clipped_fraction"], saturation_k=s.get("saturation_k"))
            for s in manifest["scenes"] if s["condition"] == "saturation"]
    return ("<section id=\"a-saturation\"><h2>A. Saturation k search</h2>"
            + full_table(pd.DataFrame(rows), "manifest.json", sources, "The search trace on the development seeds "
                         "(saturation_k).", anchor="a-saturation-trace")
            + "<h3 id=\"a-saturation-scenes\">Saturation scenes</h3>"
            + full_table(pd.DataFrame(held), "manifest.json", sources, "Achieved clipped fraction of every saturation "
                         "scene (scenes).", anchor="a-saturation-scenes") + "</section>")


def render_checks_section(verified, checks):
    rows = "".join(f"<tr><td>{code(v['key'])}</td><td>{esc(v['what'])}</td><td>{code(v['sha256'])}</td></tr>"
                   for v in verified)
    head = ["check", "condition or set", "FOV", "arm", "mode", "threshold", "detected", "matched", "correct reads",
            "wrong-gene reads", "false-detection reads"]
    detections = "".join(
        "<tr>" + "".join(f"<td>{esc(fmt(v))}</td>" for v in (
            c["what"], c.get("condition") or c.get("multi_fov"), c.get("fov_id"), c["arm"], c["threshold_mode"],
            c["threshold"], c["n_detected"], c["n_matched"], c["reads_correct"], c["reads_wrong_gene"],
            c["reads_false_detection"])) + "</tr>" for c in checks)
    return ("<section id=\"render-checks\"><h2>A. Render checks</h2><p>The evaluation saves checksums, not images. "
            "The renderer regenerated each displayed scene from its recorded configuration (the saturation k from the "
            "manifest) and preprocessed it with the recorded recipe. Every scene's per-round image checksums and every "
            "output below equal the manifest's <code>scenes</code>, <code>multi_fov_scenes</code> and "
            "<code>runs</code> records (sha256 over the rounds in codebook order); a mismatch stops rendering before "
            "anything is written.</p><div class=\"scroll\"><table class=\"fit\">" + _colgroup([330, 330, None])
            + "<thead><tr><th>condition/dtype/seed/arm[/FOV]</th>"
            f"<th>array</th><th>sha256 (matches the manifest)</th></tr></thead><tbody>{rows}</tbody></table></div>"
            "<h3 id=\"render-checks-detections\">Detections of the overlays</h3><p>Each overlay's detection ran once "
            "on the verified image at the saved "
            "development-selected value; its spot, match and read counts and its per-channel cutoffs equal the saved "
            "curve row of the same condition, seed, arm, mode and value (<code>curves/pr_curves.csv</code>, "
            "<code>curves/multi_fov_curves.csv</code>), or rendering stops.</p>"
            "<div class=\"scroll\"><table class=\"fit\">"
            + _colgroup([330] + [None] * (len(head) - 1)) + "<thead><tr>"
            + "".join(f"<th>{h}</th>" for h in head) + f"</tr></thead><tbody>{detections}</tbody></table></div>"
            "</section>")


def identity_section(manifest, manifest_path, manifest_sha, identity, rendering, sources, render_command):
    software = manifest["software"]
    run_command = " ".join(["uv run python"] + [a if " " not in a else json.dumps(a) for a in software["command"]])
    rendering_state = (f"HEAD {code(rendering['revision'])}, " + (
        f"with uncommitted changes (sha256 of <code>git diff HEAD --binary</code> plus untracked files: "
        f"{code(rendering['uncommitted_diff_sha256'])})" if rendering["dirty"] else "clean tree"))
    return f"""<section id="identity"><h2>A. Identity, revision and commands</h2>
<table><tbody>
<tr><th>Code revision (rendering)</th><td>{rendering_state}</td></tr>
<tr><th>Evaluated code revision</th><td>{code(identity['evaluated_revision'])}</td></tr>
<tr><th>Manifest revision</th><td>{code(identity['manifest_revision'])}, dirty={fmt(identity['dirty'])},
 <code>uncommitted_diff_sha256</code> {code(identity['uncommitted_diff_sha256'])}</td></tr>
<tr><th>Revision check</th><td>Passed before rendering: the manifest revision is an ancestor of HEAD;
 {esc(identity['match'])}; {', '.join(code(s) for s in identity['sources_unchanged'])} are unchanged from
 {code(identity['evaluated_revision'][:7])} to the rendering tree.</td></tr>
<tr><th>Evaluation</th><td>{code(manifest.get('issue'))}, design {code(manifest.get('design'))}, scope
 {code(manifest.get('scope'))}; {esc(manifest.get('specification', ''))}</td></tr>
<tr><th>Manifest</th><td>{code(manifest_path)}</td></tr>
<tr><th>Manifest sha256</th><td>{code(manifest_sha)}</td></tr>
<tr><th>Saved outputs</th><td>{len(sources.records) - 1} files listed in the manifest; every size and sha256 verified
 before rendering (see <a href="#sources">Sources and checksums</a>).</td></tr>
<tr><th>Software</th><td>Python {esc(software.get('python'))}, NumPy {esc(software.get('numpy'))}, pandas
 {esc(software.get('pandas'))}</td></tr>
<tr><th>Report schema</th><td>{code(CALIBRATED_REPORT_SCHEMA)}</td></tr>
</tbody></table>
<h3>Commands</h3>
<p>Evaluation (from <code>src/python</code>, as recorded in the manifest):</p><pre><code>{esc(run_command)}</code></pre>
<p>This report (from <code>src/python</code>):</p><pre><code>{esc(render_command)}</code></pre></section>"""


def capture_anchors(document):
    """The ids of the report in page order, for the browser check: one capture per id, except an id whose
    content up to the next id holds no table, figure, list or paragraph (the next capture shows it)."""
    found = [(m.start(), m.group(1)) for m in re.finditer(r"\sid=\"([^\"]+)\"", document)]
    anchors = []
    for (start, name), (end, _next) in zip(found, found[1:] + [(len(document), None)]):
        if re.search(r"<(table|img|ul|ol|dl|p|pre|details)\b", document[start:end]):
            anchors.append(name)
    return anchors


def render_calibrated(evaluation, output, *, repo=ROOT, command=None):
    """Check a calibrated evaluation (W-248) and write the item 6 report; returns a summary dict.

    Nothing is written unless every manifest-listed file matches its size and sha256, the manifest revision
    resolves to this checkout, every regenerated scene and output matches the manifest and every displayed
    detection reproduces its saved curve row.
    """
    evaluation, output = Path(evaluation).resolve(), Path(output)
    manifest_path = evaluation / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != harness.CALIBRATED_SCHEMA:
        raise SourceMismatch(f"unexpected manifest schema {manifest.get('schema')}")
    identity = check_revision(manifest["software"], repo, CALIBRATED_SOURCES)
    records = verify_sources(evaluation, manifest)
    records["manifest.json"] = dict(path="manifest.json", bytes=len(manifest_bytes), sha256=manifest_sha)
    sources = Sources(evaluation, records)
    sources.used.add("manifest.json")
    rendering = harness._revision()
    lookups = Lookups(sources)
    comparisons = sources.table("tables/comparisons.csv")
    mad_zero = {(r["condition"], r["dtype"], r["arm"]) for r in manifest.get("mad_zero_reference_round", [])}
    single_curves = sources.table("curves/pr_curves.csv")
    pooled = (harness.pool_multi_fov(sources.table("curves/multi_fov_curves.csv"))
              if "curves/multi_fov_curves.csv" in sources.records else single_curves.iloc[:0])
    figures, checks = {}, []
    with tempfile.TemporaryDirectory(prefix="w249-") as workdir:
        scenes = CalibratedScenes(manifest, workdir)
        for method, card in CARDS.items():
            before, after, _targeted = comparison_spec(method, card["comparison"])
            labels = dict(before=before, after=after + (f" ({harness.SNAPSHOT} snapshot)" if card.get("snapshot")
                                                        else ""))
            rows = card_rows(method, scenes, sources, checks)
            panels, facts = panels_figure(rows, labels)
            colour, puncta = colour_figure(rows[0], labels)
            dots, _count = dot_figure(method, comparisons)
            pr, collapsed_notes = pr_figure(method, card["comparison"], pooled if method == "sample_level_fitting"
                                            else single_curves, mad_zero)
            dropped = [f"{_row_title(r)}: {', '.join(c for i, c in enumerate(r['book'].channel_labels) if i not in signal_channels(r['truth'], r['book']))}"
                       for r in rows if len(signal_channels(r["truth"], r["book"])) < len(r["book"].channel_labels)]
            figures[method] = dict(rows=rows, labels=labels, panels=panels, facts=facts, overlays=overlay_figure(rows),
                                   colour=colour, puncta=puncta, dots=dots, pr=pr, collapsed=collapsed_notes,
                                   histograms=histogram_figure(rows, labels),
                                   background=background_figure(rows, after) if card.get("background") else None,
                                   channels_note=("channels without truth puncta, not shown: " + "; ".join(dropped))
                                   if dropped else "every channel carries truth puncta here, so none is dropped")
        verified = scenes.verified
    render_command = command or ("uv run python ../../benchmarks/preprocessing_report.py "
                                 f"--evaluation {evaluation} --output {output.resolve()}")
    cards = "".join(card_section(m, manifest, sources, lookups) for m in CARDS)
    contents = " ".join(f"<a href=\"#{a}\">{t}</a>" for a, t in (
        ("setup", "Setup"), ("cards", "Method cards"), ("findings", "Findings"), ("reading-order", "Reading order"),
        ("method-figures", "Figures"), ("appendix", "Appendix")))
    body = f"""
<h1>§2.5 preprocessing: calibrated rerun report (W-249)</h1>
<p class="banner"><strong>Development evidence on calibrated synthetic data.</strong> Rendered from the saved W-248
evaluation of the calibrated presets (W-238 development targets, <strong>not D04</strong>). It is not scientific
acceptance, it recommends <strong>no preprocessing default</strong>, and the low-benefit flag and its 0.02 threshold
are <strong>provisional</strong>. Both threshold modes are shown side by side throughout; nothing is pooled across
modes.</p>
<nav class="contents">Contents: {contents}</nav>
<section id="summary"><h2>Human summary</h2>
<p>The summary follows item 6 of the accepted amendment (the W-237 format): 1. setup, 2. one card per method or mode,
3. cross-cutting findings and anomalies, 4. reading order. Figures follow in the method sections; identity, checksums
and full tables are in the appendix.</p></section>
{setup_section(manifest, sources)}
<section id="cards"><h2>2. Method cards</h2><p>In the order of the page's comparison table. Every card gives the
targeted conditions with their precondition status, the key numbers in both modes, the flag verdict per dtype and mode
with its failing clause, and the caveats.</p>{cards}</section>
{findings_section(manifest, sources)}
{reading_order_section()}
<section id="method-figures"><h2>Method sections with figures</h2><p>The nine visualization changes of item 6, per
method, on the displayed rows of its card.</p></section>
{''.join(figures_section(m, figures[m], sources) for m in CARDS)}
<section id="appendix"><h2>Appendix</h2><p>Audit material: identity, revision and checksums, the render checks, and
every saved table in full, split by dtype where the table has one.</p></section>
{identity_section(manifest, manifest_path, manifest_sha, identity, rendering, sources, render_command)}
{render_checks_section(verified, checks)}
{projection_section(manifest, sources)}
{saturation_section(manifest, sources)}
{appendix_tables(manifest, sources)}
{sources_section(manifest_path, manifest_sha, sources)}
"""
    document = (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<title>§2.5 preprocessing: calibrated rerun report</title><style>{CALIBRATED_CSS}</style></head>"
                f"<body>{body}</body></html>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document.encode())  # one full-file write (network mounts)
    return dict(output=str(output), bytes=len(document.encode()), schema=CALIBRATED_REPORT_SCHEMA,
                manifest_sha256=manifest_sha, files_verified=len(records) - 1, identity=identity, rendering=rendering,
                verified_arrays=len(verified), detection_checks=len(checks), checks=checks, verified=verified,
                figures={m: [a for a, _t in FIGURE_ANCHORS if a != "background" or CARDS[m].get("background")]
                         for m in CARDS}, sources_used=sorted(sources.used),
                anchors=capture_anchors(document))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="W-233 or calibrated (W-248) evaluation directory (manifest.json)")
    parser.add_argument("--output", type=Path, required=True, help="report HTML file, outside Git")
    parser.add_argument("--summary", type=Path, help="also write the render summary as JSON (outside Git)")
    args = parser.parse_args()
    command = " ".join(["uv run python", os.path.relpath(Path(sys.argv[0]).resolve(), Path.cwd())]
                       + [f"--evaluation {args.evaluation}", f"--output {args.output}"]
                       + ([f"--summary {args.summary}"] if args.summary else []))
    summary = render(args.evaluation, args.output, command=command)
    text = json.dumps(summary, indent=1, default=str)
    if args.summary:
        args.summary.write_bytes((text + "\n").encode())  # one full-file write (network mounts)
    print(text)


if __name__ == "__main__":
    main()
