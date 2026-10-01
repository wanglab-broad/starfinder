# Learned detectors: install, weights, run and tests

This page collects the steps for running the two learned spot detectors of §2.7,
Spotiflow (`SpotiflowConfig`) and Piscis (`PiscisConfig`). It covers installing
their extras, fetching their pretrained weights, configuring a run and running
their tests. The rules behind these steps are in {doc}`spot-finding-contract`
("Pretrained weights", "Execution device" and "Workflow configuration"), and the
methods are described in {doc}`spot-finding-algorithms`. This page does not
recommend a method or a model. The model names in the examples are placeholders
for any row of the table below; method comparisons belong to E02.

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

* **Python versions.** Starfinder supports Python 3.10 or newer, but the extras
  carry the marker `python_version < '3.14'` because torch 2.7.1 has no wheel for
  Python 3.14. The learned detectors therefore need Python 3.10 to 3.13. On
  Python 3.14 and later the extras install nothing. A learned method then raises
  `SpotFindingBackendUnavailableError`, whose message ends with "(the extra is not
  available on Python 3.14 and later)". The recorded runs below used Python 3.12.
* **CPU-only torch.** On Linux, `[tool.uv.sources]` takes torch and torchvision from
  the explicit `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`), so the
  environment holds the CPU build `torch 2.7.1+cpu`. On other platforms uv takes
  them from PyPI. The only execution device in §2.7 is `cpu`
  (`ExecutionConfig.device` and the `device` keyword of `find_spots`, default
  `"cpu"`). Any other value raises `ValueError("device must be 'cpu'; §2.7 runs on
  CPU only")`.
* **Without the extra.** Importing `starfinder.spot_finding` and constructing a
  config never need the extras. Running a detection without its extra raises
  `SpotFindingBackendUnavailableError`, an `ImportError`, for example
  `spot-finding method 'spotiflow' requires spotiflow; install the 'spotiflow'
  extra (starfinder[spotiflow])`.

## Weights

Starfinder knows six pretrained models, listed in `KNOWN_WEIGHTS`
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
full SHA-256 values, and `KNOWN_WEIGHTS` (`starfinder.spot_finding`) holds them;
the contract's known-weights table shows abbreviated hashes.

### Commands

```bash
uv run starfinder weights list
uv run starfinder weights fetch spotiflow smfish_3d
uv run starfinder weights fetch piscis 20251212
uv run starfinder weights verify
uv run starfinder weights verify piscis 20251212
```

* `starfinder weights fetch <method> <model>` is the only step that uses the
  network. It downloads the file to a temporary name in the cache and checks its
  size and SHA-256, plus the library's MD5 for a Spotiflow archive. Only then
  does it extract the archive (Spotiflow) or move the file (Piscis) into
  `<root>/<method>/<model>/`, next to a `starfinder-weights.json` record of the
  per-file hashes. It prints the model folder.
* When the model folder already exists, `fetch` downloads nothing and never
  overwrites it. It checks only the file the library loads (`best.pt` for a
  Spotiflow model, the `.pt` file for Piscis) and that `starfinder-weights.json`
  exists, then returns the folder. It does not check or restore the other four
  files of a Spotiflow folder, so it can succeed for a folder that is missing
  `config.yaml`, for example. Run `starfinder weights verify` after a fetch to check
  every file.
* `starfinder weights list` prints the cache root and one line per model: method,
  model, dimensionality, size, SHA-256, revision and the local state. The state
  `fetched` means the folder holds `starfinder-weights.json`, `incomplete` means
  the folder exists without it, and `not fetched` means there is no folder. `list`
  hashes nothing, so `fetched` does not mean verified.
* `starfinder weights verify` re-hashes every file `KNOWN_WEIGHTS` lists for each
  known model that has a folder in the cache: five files for a Spotiflow model,
  the `.pt` file for Piscis. `starfinder weights verify <method> <model>` re-hashes
  one model. Each model prints `verified` or `failed` with the error, and the
  command exits with status 1 when any listed file is missing or changed.
  Detection runs the same full check.
* The Python equivalent of `fetch` is `fetch_weights(method, model, *,
  directory=None)`, with the same checks. `resolve_weights(method, model, *,
  directory=None, extracted=())` checks by default only the loaded file, as `fetch`
  does for an existing folder. For the full check of `verify` and of detection,
  also pass every other listed file:
  `extracted=tuple(f.path for f in KNOWN_WEIGHTS[(method, model)].extracted)`
  (empty for Piscis, whose `.pt` file is already checked).

### Cache location

Each command takes `--dir DIR`. Without it, the cache root is
`STARFINDER_WEIGHTS_DIR` when that is set, otherwise `$XDG_CACHE_HOME/starfinder/weights`,
by default `~/.cache/starfinder/weights`. A relative or `~` root is resolved to an
absolute path. The libraries' own caches (`~/.spotiflow`, `~/.piscis/models` and
the Hugging Face cache) are never read or written.

Detection has no directory option: it reads the weights from
`STARFINDER_WEIGHTS_DIR` or the default root. If you fetched with `--dir DIR`, set
`STARFINDER_WEIGHTS_DIR=DIR` before you run a detection or the tests.

### Detection never downloads

`find_spots`, `FOV.find_spots`, `FOV.run` and the workflow rules never download
anything, and no detection code path imports the fetch module. Before the model is
built, every detection checks that the model's files exist and recomputes the
SHA-256 of each one: five files for a Spotiflow model, the `.pt` file for Piscis.
A loaded model is reused within the same process.

### Errors

| Error | When | What to do |
| --- | --- | --- |
| `SpotFindingBackendUnavailableError` (an `ImportError`) | The method's extra is not installed. It is reported before any weights error. | Install the extra (see "Install"). |
| `MissingWeightsError` (a `FileNotFoundError`) | The model folder, or a file it lists, is missing. The message names the method, model, expected path and the fetch command. | If the model folder does not exist, run the fetch command it names, into the same root. If the folder exists, remove it first and then fetch: `fetch` keeps an existing folder whose loaded file verifies and does not restore missing files. Then run `starfinder weights verify`. |
| `WeightsHashMismatchError` (a `ValueError`) | A local file's SHA-256 differs from the table. The message names the file and both hashes. `fetch` raises it too when a download's size, SHA-256 or MD5 differs, or the download lacks the loaded file, and then installs nothing. | Remove the model folder and fetch again. `fetch` does not replace an existing copy. |
| `FileExistsError` | `fetch` finds a model folder whose loaded file verifies but that has no `starfinder-weights.json`. | Remove the folder and fetch again. |
| `ValueError` | The config names an unknown model, or a model of the other method. It is raised when the config is constructed, and the message lists the method's known models. A model that is empty or not a string gets "model must name a <method> model of KNOWN_WEIGHTS" instead. | Use a model from the table. |

## Run

Name the model explicitly, keep `device` at `cpu` and keep `scale` at 1. `scale`
must be 1 (the default is 1.0); any other value raises `ValueError`, because
resampling is an explicit preprocessing step.

In Python, put the config in a `PipelineConfig` and pass an `ExecutionConfig` to
`FOV.run`:

```python
from starfinder.dataset import ExecutionConfig, PipelineConfig
from starfinder.spot_finding import PiscisConfig, SpotiflowConfig

execution = ExecutionConfig(device="cpu")
spotiflow = PipelineConfig(spot_finding=SpotiflowConfig(model="smfish_3d", scale=1.0))
piscis = PipelineConfig(spot_finding=PiscisConfig(model="20251212", scale=1.0))
# Add the loading, preprocessing, registration and later stages your run needs, then:
# fov.run(spotiflow, execution=execution)
```

For one image, call `find_spots` directly:

```python
from starfinder.image import ImageMetadata
from starfinder.spot_finding import SpotiflowConfig, find_spots

result = find_spots(image, config=SpotiflowConfig(model="smfish_3d", scale=1.0),
                    metadata=ImageMetadata("fov001"), spot_namespace="dataset/fov001", device="cpu")
```

In the workflow YAML, the method keys are Python-only, so the configuration needs
`backend: python`. The five Python rules accept the `spot_finding` block, and the
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
        scale: 1
```

For Piscis, use `method: piscis` and a Piscis model:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      device: cpu
      spot_finding:
        run: true
        ref_round: round1
        method: piscis
        model: "20251212"
        scale: 1
```

Quote a Piscis model name in YAML: an unquoted `20251212` is read as an integer.
The other fields (Spotiflow `prob_thresh`, `min_distance`, `exclude_border`,
`subpix` and `n_tiles`; Piscis `threshold`, `min_distance` and `input_size`) keep
their defaults when omitted. See {doc}`workflow-configuration` ("Spot-finding
method key") for `channel_overrides` and `rounds`.

## Tests

Tests are in `src/python/test`. The default tier deselects the `extended` mark
(`addopts` in `pyproject.toml`), so `uv run pytest test/ -v` never runs a learned
detector.

| Module | Tier | Needs |
| --- | --- | --- |
| `test_spot_finding_weights.py` | default | Neither extra nor weights. Fixture entries are fetched through `file://` URLs, and sockets are blocked. |
| `test_spot_finding_learned.py` | default | Neither extra nor weights. The extras are simulated missing, or present as empty stand-ins. |
| `test_spotiflow.py` | extended | The `spotiflow` extra and the four Spotiflow models |
| `test_piscis.py` | extended | The `piscis` extra and both Piscis models |
| `test_spot_finding_validation.py` | Local maxima and Starfish LoG cases: default. The `spotiflow-<model>` and `piscis-<model>` cases: extended. | The extra and the model of each learned case |

The learned tests read the weights from `STARFINDER_WEIGHTS_DIR` or the default
cache root. Without the extra, a learned test is skipped (`pytest.importorskip`).
With the extra installed but the weights not fetched, it fails with
`MissingWeightsError`, which names the fetch command.

The GitHub CI workflow (`.github/workflows/tests.yml`) installs the `dev`,
`local-registration` and `checkpoint` extras on Python 3.12, and neither learned
extra. It runs the default and the extended suites, so the learned tests are
skipped there and nothing is downloaded. Run them locally.

From `src/python`, after installing the extras and fetching the six models, run:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
uv run starfinder weights verify
uv run pytest test/test_spotiflow.py -v -m extended
uv run pytest test/test_piscis.py -v -m extended
uv run pytest test/ -v -m extended -k "test_spot_finding_validation and not (20230905 or 20251212)"
uv run pytest test/ -v -m extended -k "test_spot_finding_validation and 20230905"
uv run pytest test/ -v -m extended -k "test_spot_finding_validation and 20251212"
```

The third command runs the Spotiflow cases of the validation module, and the last
two run its Piscis cases, one model at a time. Starfinder does not change thread
settings: the exports give one numerical thread and no GPU, as in the recorded
runs, and the extended tests also set them in the test process. Some validation
cases are strict expected failures recorded
in W-274, and pytest reports them as `xfailed`.

Approximate run times:

| Selection | Approximate time | Recorded run |
| --- | --- | --- |
| `test_spotiflow.py -m extended` | 1.5 minutes | 1:21 wall clock, 42 passed (W-272) |
| `test_piscis.py -m extended` | 9 minutes | 8:42 wall clock, 44 passed (W-272) |
| Validation, Spotiflow cases | 3 minutes | about 165 s of test time (W-274) |
| Validation, `20230905` cases | 18 minutes | 17:42 wall clock, 58 passed, 5 xfailed (W-274 check) |
| Validation, `20251212` cases | 18 minutes | 17:46 wall clock, 59 passed, 5 xfailed (W-274 check) |
| The extended tier without the two Piscis validation selections | 16 minutes | 15:59 wall clock, 248 passed, 4 xfailed (W-274 check) |

These times come from single recorded runs in the W-272 and W-274 records; they
were not measured again for this page. They are approximate one-thread values
from one host (GP099-29C, Linux, Python 3.12, torch 2.7.1+cpu), measured with
`taskset -c 0`, the thread variables at 1 and no GPU. They will differ on other
machines and as the tests change.
