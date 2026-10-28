# Learned detectors: install, weights and run

Spot finding has two learned detectors, Spotiflow (`SpotiflowConfig`) and Piscis
(`PiscisConfig`). This page shows how to install their extras, fetch their pretrained
weights and run them. The {doc}`spot-finding guide <spot-finding-contract>` describes
the stage, and the {doc}`spot-finding reference <spot-finding-algorithms>` lists each
config's options. This page does not recommend a method or a model; the model names
in the examples stand for any row of the table below.

## Install

Each detector has its own optional extra in `src/python/pyproject.toml`:

| Extra | Installs |
| --- | --- |
| `spotiflow` | spotiflow 0.6.5, torch 2.7.1 and torchvision 0.22.1 |
| `piscis` | piscis 1.1.0, torch 2.7.1 and torchvision 0.22.1 |

Run these commands from `src/python`, naming one or both extras:

```bash
cd src/python
uv sync --locked --extra spotiflow --extra piscis
```

`uv sync` installs only the extras it is given. Name every extra you need, for
example `--extra local-registration`, on the same command and on every later
`uv sync`. Run the commands below with `uv run` from `src/python`.

* **Python versions.** The extras carry the marker `python_version < '3.14'`, because
  torch 2.7.1 has no wheel for Python 3.14. The learned detectors therefore need
  Python 3.11 to 3.13. On Python 3.14 and later the extras install nothing, and a
  learned method raises `SpotFindingBackendUnavailableError`, whose message ends with
  "(the extra is not available on Python 3.14 and later)".
* **CPU-only torch.** On Linux, uv takes torch and torchvision from the PyTorch CPU
  index, so the environment holds `torch 2.7.1+cpu`. On other platforms uv takes them
  from PyPI. Spot finding runs on the CPU only.
* **Without the extra.** Importing `starfinder.spot_finding` and constructing a
  config never need the extras. Running a detection without its extra raises
  `SpotFindingBackendUnavailableError`, an `ImportError`, for example
  `spot-finding method 'spotiflow' requires spotiflow; install the 'spotiflow'
  extra (starfinder[spotiflow])`.

## Weights

Starfinder knows six pretrained models, listed in `KNOWN_MODELS`
(`starfinder.spot_finding`). There is no default model: every config names one.

| Method | Model | Network | Images it detects | Download (bytes) | Source |
| --- | --- | --- | --- | --- | --- |
| `spotiflow` | `synth_3d` | 3D | Z>1, in 3D | 263,107,693 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `smfish_3d` | 3D | Z>1, in 3D | 263,106,200 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `general` | 2D | Z=1, as a YX plane | 87,885,382 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `hybiss` | 2D | Z=1, as a YX plane | 87,945,684 | `spotiflow-models` release 0.6.0 |
| `piscis` | `20230905` | 2D | Z=1 in plane mode, Z>1 in stack mode | 30,077,822 | Hugging Face `wniu/Piscis` at `9bdefc72cb` |
| `piscis` | `20251212` | 2D | Z=1 in plane mode, Z>1 in stack mode | 30,143,014 | Hugging Face `wniu/Piscis` at `9bdefc72cb` |

A Spotiflow model with the other dimensionality, or a shape below the model's
minimum, raises `IncompatibleGeometryError`. `starfinder weights list` prints the
full SHA-256 values, and `KNOWN_MODELS` holds them.

### Commands

```bash
uv run starfinder weights list
uv run starfinder weights fetch spotiflow smfish_3d
uv run starfinder weights fetch piscis 20251212
uv run starfinder weights verify
uv run starfinder weights verify piscis 20251212
```

* `starfinder weights fetch <method> <model>` is the only step that uses the
  network. It downloads the file, checks its size and SHA-256 (and the library's MD5
  for a Spotiflow archive) and every file `KNOWN_MODELS` lists for the model, and only
  then installs the model in `<root>/<method>/<model>/`, next to a
  `starfinder-weights.json` record of the per-file hashes. It prints the model folder.
  It never overwrites a file in an existing model folder; the
  [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md#fetching-cache-and-verification)
  says what it does with one.
* `starfinder weights list` prints the cache root and one line per model: method,
  model, dimensionality, size, SHA-256, revision and the local state. The state
  `fetched` means the folder holds `starfinder-weights.json`, `incomplete` means
  the folder exists without it, and `not fetched` means there is no folder. `list`
  hashes nothing, so `fetched` does not mean verified.
* `starfinder weights verify` re-hashes every file `KNOWN_MODELS` lists for each
  known model that has a folder in the cache: five files for a Spotiflow model,
  the `.pt` file for Piscis. `starfinder weights verify <method> <model>` re-hashes
  one model. Each model prints `verified` or `failed` with the error, and the
  command exits with status 1 when any listed file is missing or changed.
* The Python equivalents are `fetch_model(method, model, *, directory=None)` and
  `resolve_model`; {py:func}`~starfinder.spot_finding.resolve_model` says which files
  it checks.
* The segmentation models share the cache and the three commands
  ({doc}`segmentation-contract`, "Models").

### Cache location

Each command takes `--dir DIR`. Without it, the cache root is
`STARFINDER_WEIGHTS_DIR` when that is set, otherwise `$XDG_CACHE_HOME/starfinder/weights`,
by default `~/.cache/starfinder/weights`. A relative or `~` root is resolved to an
absolute path. The libraries' own caches (`~/.spotiflow`, `~/.piscis/models` and
the Hugging Face cache) are never read or written.

Detection has no directory option: it reads the weights from
`STARFINDER_WEIGHTS_DIR` or the default root. If you fetched with `--dir DIR`, set
`STARFINDER_WEIGHTS_DIR=DIR` before you run a detection.

### Detection never downloads

`find_spots`, `FOV.find_spots`, `FOV.run` and the workflow rules never download
anything. Before the model is built, every detection checks that the model's files
exist and recomputes the SHA-256 of each one: five files for a Spotiflow model, the
`.pt` file for Piscis. A loaded model is reused within the same process.

### Errors

| Error | When | What to do |
| --- | --- | --- |
| `SpotFindingBackendUnavailableError` (an `ImportError`) | The method's extra is not installed. It is reported before any weights error. | Install the extra (see "Install"). |
| `MissingModelError` (a `FileNotFoundError`) | The model folder, or a file it lists, is missing. The message names the method, model, expected path and the fetch command. | Run the fetch command it names, into the same root, then `starfinder weights verify`. |
| `ModelHashMismatchError` (a `ValueError`) | A file in the cache or in a download differs from the table, or a download lacks a listed file. The message names the file. | Remove the model folder and fetch again. |
| `FileExistsError` | `fetch` finds a model folder whose present files verify but that has no `starfinder-weights.json`. | Remove the folder and fetch again. |
| `ValueError` | The config names an unknown model, or a model of the other method, when it is constructed. The message lists the method's known models. | Use a model from the table. |

## Run

Name the model explicitly, and keep `device` at `cpu` and `scale` at 1, their
defaults; other values raise `ValueError`.

In Python, put the config in a `PipelineConfig` and pass an `ExecutionConfig` to
`FOV.run`:

```python
from starfinder.dataset import ExecutionConfig, PipelineConfig
from starfinder.spot_finding import PiscisConfig, SpotiflowConfig

execution = ExecutionConfig(device="cpu")
spotiflow = PipelineConfig(spot_finding=SpotiflowConfig(model="smfish_3d"))
piscis = PipelineConfig(spot_finding=PiscisConfig(model="20251212"))
# Add the loading, preprocessing, registration and later stages your run needs, then:
# fov.run(spotiflow, execution=execution)
```

For one image, call `find_spots` directly:

```python
from starfinder.image import ImageMetadata
from starfinder.spot_finding import SpotiflowConfig, find_spots

result = find_spots(image, config=SpotiflowConfig(model="smfish_3d"),
                    metadata=ImageMetadata("fov001"), spot_namespace="dataset/fov001", device="cpu")
```

In the workflow YAML, the method keys are Python-only, so the configuration needs
`backend: python`. The Python rules accept the `spot_finding` block, and the
rule-level `device` key sets `ExecutionConfig.device`:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      device: cpu
      spot_finding:
        run: true
        ref_round: round1
        method: spotiflow
        model: smfish_3d
```

For Piscis, use `method: piscis` and a Piscis model, quoted: an unquoted `20251212`
is read as an integer, for example `model: "20251212"`. The other fields keep their
defaults when omitted. See {doc}`workflow-configuration` ("Spot-finding method key")
for `channel_overrides` and `rounds`.

Starfinder does not change thread settings. To run on one thread, set
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `NUMBA_NUM_THREADS` likewise, before you start Python.

## Tests

The GitHub CI workflow installs neither learned extra and deselects the tests marked
`learned`. A test that runs Spotiflow or Piscis is skipped when the extra is not
installed, and fails with `MissingModelError` when the extra is installed but the
weights are not fetched. The test modules, their
tiers and the commands that run them are in
[`design/notes/learned-detectors-tests.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/learned-detectors-tests.md).
